#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Demucs 服务（桌面加速模式）

让网页把耗时的音轨分离计算交给本机运行（CPU/GPU 都行），
浏览器只负责上传文件、轮询进度、下载并播放/保存结果。

这是为了解决「300MB 的 htdemucs ONNX 模型在 ONNX Runtime Web 的 wasm 后端上
session.run() 必崩」这个浏览器硬限制：真正的推理放到本机 Python 里跑，
既能用完整版 Demucs、又能用 GPU，还不受浏览器内存/wasm 限制。

启动：
    pip install -r requirements.txt
    python server.py                 # 默认 http://127.0.0.1:8001
    python server.py --port 8001 --model htdemucs --device cuda

接口（详见 README「本地服务模式」）：
    GET  /api/health                       -> {"status","model","device"}
    POST /api/separate  (multipart: file) -> {"job_id": "..."}  （后台异步处理）
    GET  /api/job/<id>                     -> {"status","progress","message","stems","subtitles","sample_rate","error"}
    GET  /api/stem/<id>/<name>             -> WAV 文件（name ∈ vocals/drums/bass/other/accompaniment）
    GET  /api/subtitle/<id>                -> 旁白字幕 SRT（分开时传 with_subtitles=1 才会生成）
"""

import argparse
import os

# Hugging Face 相关默认值（可在启动前用环境变量覆盖）：
# 1) 国内访问 huggingface.co 不稳，默认走镜像；
# 2) hf-mirror 不支持 Xet 存储协议，必须禁用 Xet 回退普通下载，
#    否则会报 “CAS Client Error: 401 Unauthorized … xethub.hf.co”。
# 这两项必须在 huggingface_hub 被 import 之前设置（faster-whisper 是懒加载，满足）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 3) 把 HF 缓存放到项目内 .cache/hf，避免某些环境 ~/.cache 无法写入。
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, ".cache", "hf"))

import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
# 允许本机网页跨源调用：静态页面可能跑在 8000，而本服务在 8001（不同源）。
CORS(app)

# ---------------- 全局配置（由命令行参数初始化）----------------
MODEL_NAME = "htdemucs"
DEVICE = "cpu"

# 任务存储：job_id -> dict(dir, status, progress, message, stems, sample_rate, error, created)
jobs = {}
_jobs_lock = threading.Lock()

# Demucs 模型单例（避免每次请求都重新加载）
_model = None
_model_lock = threading.Lock()

# Whisper（语音识别）模型单例，用于给人声生成带时间戳的旁白字幕（SRT）
_asr_model = None
_asr_lock = threading.Lock()
ASR_MODEL_NAME = "small"   # tiny/base/small/medium/large-v3
ASR_LANGUAGE = None        # None = 自动识别语言；"zh" 可强制中文

# 临时目录里允许下载的音轨名
ALLOWED_STEMS = ("vocals", "drums", "bass", "other", "accompaniment")

# 临时目录最多保留多久（秒），超时在下次提交任务时清理
JOB_TTL = 60 * 60


def get_model():
    """懒加载并缓存 Demucs 模型（线程安全）。兼容 demucs 3.x/4.x 的函数命名。"""
    global _model
    with _model_lock:
        if _model is None:
            # demucs 4.x 只导出 get_model；更早版本只导出 load_model。逐个尝试，避免
            # “from ... import a, b” 在 b 不存在时直接抛 ImportError。
            try:
                from demucs.pretrained import get_model as loader
            except ImportError:
                from demucs.pretrained import load_model as loader

            m = loader(MODEL_NAME)
            m.eval()
            m.to(DEVICE)
            _model = m
        return _model


def get_asr_model():
    """懒加载并缓存 faster-whisper 模型（线程安全）。

    faster-whisper 用 CTranslate2 推理，CPU 也很流畅，且自带音频解码（PyAV），
    不依赖系统 ffmpeg。首次会从 Hugging Face 下载模型权重。
    """
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            from faster_whisper import WhisperModel  # 可能未安装

            compute_type = "float16" if DEVICE == "cuda" else "int8"
            _asr_model = WhisperModel(
                ASR_MODEL_NAME,
                device="cuda" if DEVICE == "cuda" else "cpu",
                compute_type=compute_type,
            )
        return _asr_model


def _srt_timestamp(seconds):
    """把秒数格式化成 SRT 时间戳：HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600 * 1000)
    m, ms = divmod(ms, 60 * 1000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_audio_for_asr(path, target_sr=16000):
    """读音频并转成 Whisper 需要的单声道 16kHz float32 numpy 数组。

    直接传 numpy 给 faster-whisper，可绕开它对 PyAV（av.open）的解码依赖，
    避免不同 PyAV 版本缺少 metadata_errors 参数导致的 TypeError。
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(path, always_2d=True)  # [frames, channels] float64
    mono = torch.from_numpy(data).float().mean(dim=1)  # -> [frames] 单声道
    if sr and sr != target_sr and mono.numel() > 0:
        new_len = max(1, int(round(mono.numel() * target_sr / sr)))
        mono = torch.nn.functional.interpolate(
            mono.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
        ).view(-1)
    return mono.numpy().astype(np.float32)


def transcribe_to_srt(wav_path, out_srt_path):
    """用 Whisper 转录音频，写成 SRT 字幕文件；返回字幕段数。"""
    model = get_asr_model()
    audio = load_audio_for_asr(wav_path)
    segments, _info = model.transcribe(
        audio,
        language=ASR_LANGUAGE,
        vad_filter=False,         # 不依赖 onnxruntime 的 VAD，交给 Whisper 自身切句
        beam_size=5,
    )

    lines = []
    index = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        index += 1
        lines.append(str(index))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")

    # utf-8-sig（带 BOM）能让 Windows 播放器/剪辑软件更可靠地识别中文
    with open(out_srt_path, "w", encoding="utf-8-sig") as fp:
        fp.write("\n".join(lines))
    return index


def update_job(job_id, **kwargs):
    with _jobs_lock:
        j = jobs.get(job_id)
        if j is None:
            return
        j.update(kwargs)


def cleanup_old_jobs():
    """删除过期任务目录，避免磁盘被临时 WAV 占满。"""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in jobs.items() if now - j.get("created", 0) > JOB_TTL]
        for jid in expired:
            d = jobs[jid].get("dir")
            if d and os.path.isdir(d):
                try:
                    import shutil

                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
            jobs.pop(jid, None)


def load_audio(input_path):
    """用 soundfile 读取音频，返回 (wav, sr)，wav 形状 [channels, samples] float32。

    soundfile（libsndfile）原生支持 WAV / FLAC / OGG 等；对 MP3 / M4A 等格式，
    若系统装了 ffmpeg，会由 load_audio_ffmpeg 兜底。
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(input_path, always_2d=True)  # [frames, channels]，float64
    wav = torch.from_numpy(data.T).float()  # -> [channels, frames]
    return wav, int(sr)


def load_audio_ffmpeg(input_path, samplerate, channels):
    """用系统 ffmpeg 把音频解码成指定采样率/声道的 float32 PCM。无 ffmpeg 时抛 FileNotFoundError。"""
    import subprocess
    import numpy as np
    import torch

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "f32le", "-ac", str(channels), "-ar", str(samplerate), "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    arr = np.frombuffer(raw, dtype=np.float32)
    frames = len(arr) // channels
    wav = torch.from_numpy(arr.reshape(channels, frames)).float()
    return wav, samplerate


def save_wav(path, tensor, samplerate):
    """用标准库 wave 把 float32 张量 [channels, samples] 写成 16-bit PCM WAV，无第三方依赖。"""
    import torch
    import wave

    channels, samples = tensor.shape
    data = tensor.clamp(-1.0, 1.0).mul(32767).to(torch.int16).cpu().numpy()
    interleaved = data.T.reshape(-1).tobytes()  # [samples, channels] 交错
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(samplerate))
        w.writeframes(interleaved)


def process_job(job_id, input_path, with_subtitles=False):
    """后台线程：用 Demucs 分离音轨并写出 WAV；可选用 Whisper 生成旁白字幕。"""
    try:
        update_job(job_id, status="processing", progress=10, message="正在加载模型...")
        model = get_model()

        update_job(job_id, progress=25, message="正在解码音频...")
        import torch
        from demucs.audio import convert_audio

        # 先试 soundfile，失败（如 MP3/M4A）再试 ffmpeg（系统需已安装）
        try:
            wav, sr = load_audio(input_path)
        except Exception:
            wav, sr = load_audio_ffmpeg(
                input_path, model.samplerate, model.audio_channels
            )
        wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

        update_job(
            job_id,
            progress=45,
            message="正在分离音轨（本地 Demucs 推理，长音频可能需要几分钟）...",
        )
        with torch.no_grad():
            # apply_model 输入 [batch, channels, length]，输出 [batch, sources, channels, length]
            sources = apply_model_safe(model, wav.unsqueeze(0), DEVICE)[0]

        stems = {name: sources[i] for i, name in enumerate(model.sources)}

        # 伴奏 = 原曲 - 人声（与网页端既有约定一致）
        accompaniment = wav - stems["vocals"]
        stems["accompaniment"] = accompaniment

        update_job(job_id, progress=85, message="正在编码 WAV 文件...")
        out_dir = Path(jobs[job_id]["dir"])
        stem_urls = {}
        for name, tensor in stems.items():
            path = out_dir / f"{name}.wav"
            save_wav(str(path), tensor.cpu(), model.samplerate)
            stem_urls[name] = f"/api/stem/{job_id}/{name}"

        # 可选：对人声做语音识别，生成带时间线的旁白字幕（SRT）
        subtitles_url = None
        subtitles_error = None
        if with_subtitles:
            try:
                update_job(
                    job_id,
                    progress=90,
                    message="正在用 Whisper 生成旁白字幕（首次需下载识别模型，可能较慢）...",
                )
                srt_path = out_dir / "subtitles.srt"
                count = transcribe_to_srt(str(out_dir / "vocals.wav"), str(srt_path))
                if count > 0:
                    subtitles_url = f"/api/subtitle/{job_id}"
                else:
                    subtitles_error = "Whisper 未识别到任何语音内容（可能是纯音乐/未检测到人声）"
            except Exception as asr_exc:  # 字幕失败不应影响已完成的音轨
                subtitles_error = str(asr_exc)
                traceback.print_exc()

        final_msg = "分离完成" if not subtitles_url else "分离完成，旁白字幕已生成"
        if subtitles_error:
            final_msg = "分离完成（字幕生成失败，详见 subtitles_error）"

        update_job(
            job_id,
            status="done",
            progress=100,
            message=final_msg,
            stems=stem_urls,
            subtitles=subtitles_url,
            subtitles_error=subtitles_error,
            sample_rate=model.samplerate,
        )
    except Exception as exc:  # noqa: BLE001
        update_job(
            job_id,
            status="error",
            progress=100,
            message=str(exc),
            error=traceback.format_exc(),
        )


def apply_model_safe(model, mix, device):
    """调用 demucs.apply.apply_model，兼容不同 demucs 版本。"""
    from demucs.apply import apply_model

    try:
        return apply_model(model, mix, device=device, progress=False, split=True, overlap=0.25)
    except TypeError:
        # 老版本没有 split/overlap 关键字
        return apply_model(model, mix, device=device, progress=False)


# ------------------------------ 路由 ------------------------------
@app.route("/api/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, device=DEVICE)


@app.route("/api/separate", methods=["POST"])
def separate():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="缺少音频文件（表单字段应为 file）"), 400

    # 是否生成旁白字幕：前端传 with_subtitles=1/true
    with_subtitles = str(request.form.get("with_subtitles", "")).lower() in (
        "1", "true", "yes", "on"
    )

    cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    d = tempfile.mkdtemp(prefix="vocalsep_")
    ext = os.path.splitext(f.filename)[1] or ".tmp"
    input_path = os.path.join(d, "input" + ext)
    f.save(input_path)

    with _jobs_lock:
        jobs[job_id] = {
            "dir": d,
            "status": "queued",
            "progress": 0,
            "message": "已接收文件，排队中...",
            "stems": None,
            "subtitles": None,
            "subtitles_error": None,
            "sample_rate": None,
            "error": None,
            "created": time.time(),
        }

    threading.Thread(
        target=process_job,
        args=(job_id, input_path, with_subtitles),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        j = jobs.get(job_id)
    if j is None:
        return jsonify(error="未知的任务 ID"), 404
    return jsonify(
        status=j["status"],
        progress=j["progress"],
        message=j["message"],
        stems=j["stems"],
        subtitles=j.get("subtitles"),
        subtitles_error=j.get("subtitles_error"),
        sample_rate=j["sample_rate"],
        error=j["error"],
    )


@app.route("/api/subtitle/<job_id>")
def subtitle_file(job_id):
    with _jobs_lock:
        j = jobs.get(job_id)
    if j is None or not j.get("subtitles"):
        return jsonify(error="该任务没有旁白字幕"), 404
    path = os.path.join(j["dir"], "subtitles.srt")
    if not os.path.exists(path):
        return jsonify(error="字幕文件尚未生成"), 404
    return send_file(
        path,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="subtitles.srt",
    )


@app.route("/api/stem/<job_id>/<name>")
def stem_file(job_id, name):
    if name not in ALLOWED_STEMS:
        return jsonify(error="非法的音轨名"), 400
    with _jobs_lock:
        j = jobs.get(job_id)
    if j is None or j["stems"] is None:
        return jsonify(error="任务不存在或尚未完成"), 404
    path = os.path.join(j["dir"], name + ".wav")
    if not os.path.exists(path):
        return jsonify(error="音轨尚未生成"), 404
    return send_file(path, mimetype="audio/wav", as_attachment=False, download_name=f"{name}.wav")


def main():
    global MODEL_NAME, DEVICE, ASR_MODEL_NAME, ASR_LANGUAGE
    parser = argparse.ArgumentParser(description="本地 Demucs 服务（供网页调用）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--model", default="htdemucs", help="Demucs 模型名，如 htdemucs / htdemucs_ft")
    parser.add_argument(
        "--device",
        default="cuda" if _torch_cuda_available() else "cpu",
        help="推理设备：cuda / cpu",
    )
    parser.add_argument(
        "--asr-model",
        default="small",
        help="Whisper 模型大小：tiny/base/small/medium/large-v3（越大越准也越慢，默认 small）",
    )
    parser.add_argument(
        "--asr-language",
        default=None,
        help="强制识别语言（如 zh）；不填则自动检测",
    )
    args = parser.parse_args()

    # 尽早检查 torch：导入失败时给出真实原因，
    # 而不是让 demucs 内部二次 import 时抛出误导性的
    # “torch can only be initialized once per process”。
    try:
        import torch  # noqa: F401

        print(f"[vocal-separator] torch 导入成功，版本 {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        print("[错误] torch 导入失败，真实原因如下：")
        print(exc)
        print()
        print("常见原因：Python 版本过新（如 3.14），torch 尚不支持。")
        print("解决办法：改用 Python 3.11~3.13 建虚拟环境，例如：")
        print("    py -3.12 -m venv .venv")
        print("    .venv\\Scripts\\pip install -r requirements.txt")
        print("    .venv\\Scripts\\python server.py")
        raise SystemExit(1)

    MODEL_NAME = args.model
    DEVICE = args.device
    ASR_MODEL_NAME = args.asr_model
    ASR_LANGUAGE = args.asr_language or None

    print(f"[vocal-separator] 本地 Demucs 服务启动：http://{args.host}:{args.port}")
    print(f"[vocal-separator] 模型={MODEL_NAME}  设备={DEVICE}")
    print(f"[vocal-separator] 旁白字幕 Whisper={ASR_MODEL_NAME}  语言={ASR_LANGUAGE or 'auto'}")
    print("[vocal-separator] 在网页里点击「开始分离」即可自动调用本服务。")
    app.run(host=args.host, port=args.port, threaded=True)


def _torch_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
