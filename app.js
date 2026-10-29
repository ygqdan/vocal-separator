/**
 * 音乐人声分离器 - 主应用逻辑
 * 基于 ONNX Runtime Web 在浏览器端运行 AI 模型
 *
 * ⚠️ 关于 ONNX Runtime 的引入方式（不要改回顶层静态 import）：
 *   1. `dist/ort.min.js` 是 UMD/IIFE 构建，不是 ES Module。
 *      写成顶层 `import * as ort from '.../dist/ort.min.js'` 时命名空间是空的，
 *      `ort.env` 为 undefined，模块求值阶段就会抛 TypeError，
 *      后面所有 addEventListener 都不会执行 —— 表现就是“选完文件毫无反应”。
 *   2. 顶层静态 import 一旦 CDN 不可达（国内访问 jsDelivr 经常失败），
 *      整个模块直接不执行，页面同样毫无反应。
 * 因此这里改为「运行时动态加载 + 多 CDN 降级」，加载失败也不影响页面交互。
 */

const ORT_VERSION = '1.17.0';
const ORT_LOAD_TIMEOUT = 15000;

// 按顺序尝试，命中即用（jsDelivr 优先，国内镜像兜底）
const ORT_CDN_BASES = [
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
    `https://fastly.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
    `https://unpkg.com/onnxruntime-web@${ORT_VERSION}/dist/`,
    `https://registry.npmmirror.com/onnxruntime-web/${ORT_VERSION}/files/dist/`
];

let ortPromise = null;
// 加载成功后的 ONNX Runtime 命名空间（供 separateAudio 里 new ort.Tensor 使用）
let ort = null;

// 动态 import 可能长时间挂起，加超时避免一直卡住
function importWithTimeout(url, timeout = ORT_LOAD_TIMEOUT) {
    return Promise.race([
        import(/* webpackIgnore: true */ url),
        new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`加载超时: ${url}`)), timeout);
        })
    ]);
}

// 注入经典 <script>（UMD 构建会挂到 window.ort 上）
function loadScriptOnce(url, timeout = ORT_LOAD_TIMEOUT) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        let settled = false;

        const finish = (error) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            if (error) {
                script.remove();
                reject(error);
            } else {
                resolve();
            }
        };

        const timer = setTimeout(() => finish(new Error(`加载超时: ${url}`)), timeout);
        script.src = url;
        script.async = true;
        script.onload = () => finish();
        script.onerror = () => finish(new Error(`加载失败: ${url}`));
        document.head.appendChild(script);
    });
}

function isValidOrt(ort) {
    return !!ort && typeof ort.InferenceSession === 'function' && !!ort.env;
}

async function loadOrt() {
    if (!ortPromise) {
        ortPromise = (async () => {
            const errors = [];

            for (const base of ORT_CDN_BASES) {
                // 优先 ESM 构建（dist/esm/ort.min.js 才是真正的 ES Module）
                try {
                    const mod = await importWithTimeout(`${base}esm/ort.min.js`);
                    // 该构建同时有 default 和具名导出，择优取可用的那个
                    const runtime = isValidOrt(mod && mod.default) ? mod.default : mod;
                    if (!isValidOrt(runtime)) {
                        throw new Error('ESM 构建未导出 InferenceSession');
                    }
                    runtime.env.wasm.wasmPaths = base;
                    ort = runtime;
                    return runtime;
                } catch (error) {
                    errors.push(`ESM ${base} → ${error.message}`);
                }

                // 回退 UMD 构建
                try {
                    delete window.ort;
                    await loadScriptOnce(`${base}ort.min.js`);
                    const runtime = window.ort;
                    if (!isValidOrt(runtime)) {
                        throw new Error('UMD 构建未挂载 window.ort');
                    }
                    runtime.env.wasm.wasmPaths = base;
                    ort = runtime;
                    return runtime;
                } catch (error) {
                    errors.push(`UMD ${base} → ${error.message}`);
                }
            }

            console.error('ONNX Runtime 所有 CDN 均加载失败:\n' + errors.join('\n'));
            throw new Error('无法加载 ONNX Runtime（所有 CDN 均不可用），请检查网络或关闭广告拦截插件');
        })();

        // 失败后清空缓存，允许用户重试
        ortPromise.catch(() => {
            ortPromise = null;
        });
    }
    return ortPromise;
}

// 模型配置
// 真实可用的 ONNX 模型：StemSplitio/htdemucs-onnx（MIT）
//   输入 mix   : [1, 2, 343980]  float32  44.1kHz 立体声，7.8 秒片段
//   输出 stems : [1, 4, 2, 343980] float32，顺序为 [drums, bass, other, vocals]
// 注意 segmentLength 必须和模型固定输入长度一致（343980 = 7.8s * 44100），不能随便改。
const MODEL_CONFIGS = {
    'htdemucs': {
        name: 'HTDemucs 完整版',
        size: '316MB',
        urls: [
            'https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx',
            'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx'
        ],
        sources: ['drums', 'bass', 'other', 'vocals'],
        inputName: 'mix',
        outputName: 'stems',
        sampleRate: 44100,
        segmentLength: 343980,
        overlap: 0.25
    },
    'htdemucs-fp16': {
        name: 'HTDemucs 轻量版',
        size: '166MB',
        // fp16 存储权重的版本：体积更小，但依赖 float16 算子支持，兼容性略差
        urls: [
            'https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx',
            'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx'
        ],
        sources: ['drums', 'bass', 'other', 'vocals'],
        inputName: 'mix',
        outputName: 'stems',
        sampleRate: 44100,
        segmentLength: 343980,
        overlap: 0.25
    }
};

// 状态管理
const state = {
    audioFile: null,
    audioBuffer: null,
    selectedModel: 'htdemucs',
    session: null,
    inputName: 'mix',
    outputName: 'stems',
    localModelData: null,   // 用户手动选择的本地 .onnx 文件
    modelLoaded: false,
    isProcessing: false,
    results: {}
};

// DOM 元素
const elements = {};

// 初始化 DOM 引用
function initElements() {
    elements.dropZone = document.getElementById('dropZone');
    elements.audioInput = document.getElementById('audioInput');
    elements.fileInfo = document.getElementById('fileInfo');
    elements.fileName = document.getElementById('fileName');
    elements.fileSize = document.getElementById('fileSize');
    elements.removeFile = document.getElementById('removeFile');
    elements.modelProgress = document.getElementById('modelProgress');
    elements.modelProgressFill = document.getElementById('modelProgressFill');
    elements.modelProgressText = document.getElementById('modelProgressText');
    elements.localModelInput = document.getElementById('localModelInput');
    elements.localModelName = document.getElementById('localModelName');
    elements.processBtn = document.getElementById('processBtn');
    elements.processProgress = document.getElementById('processProgress');
    elements.processBar = document.getElementById('processBar');
    elements.processStage = document.getElementById('processStage');
    elements.processPercent = document.getElementById('processPercent');
    elements.processProgressFill = document.getElementById('processProgressFill');
    elements.processProgressText = document.getElementById('processProgressText');
    elements.processedTime = document.getElementById('processedTime');
    elements.totalTime = document.getElementById('totalTime');
    elements.resultsSection = document.getElementById('resultsSection');
    elements.vocalAudio = document.getElementById('vocalAudio');
    elements.accompanimentAudio = document.getElementById('accompanimentAudio');
    elements.drumsAudio = document.getElementById('drumsAudio');
    elements.bassAudio = document.getElementById('bassAudio');
    elements.otherAudio = document.getElementById('otherAudio');
    elements.vocalDownload = document.getElementById('vocalDownload');
    elements.accompanimentDownload = document.getElementById('accompanimentDownload');
    elements.drumsDownload = document.getElementById('drumsDownload');
    elements.bassDownload = document.getElementById('bassDownload');
    elements.otherDownload = document.getElementById('otherDownload');
    elements.vocalDuration = document.getElementById('vocalDuration');
    elements.accompanimentDuration = document.getElementById('accompanimentDuration');
    elements.drumsDuration = document.getElementById('drumsDuration');
    elements.bassDuration = document.getElementById('bassDuration');
    elements.otherDuration = document.getElementById('otherDuration');
    elements.drumsCard = document.getElementById('drumsCard');
    elements.bassCard = document.getElementById('bassCard');
    elements.otherCard = document.getElementById('otherCard');
    elements.processAnother = document.getElementById('processAnother');

    // 模型选择单选按钮
    elements.modelRadios = document.querySelectorAll('input[name="model"]');
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

// 格式化时间
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// 显示通知
function showToast(message, type = 'info') {
    // 简单的 toast 实现
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease;
        background: ${type === 'error' ? '#dc2626' : type === 'success' ? '#059669' : '#2563eb'};
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// 添加 toast 动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// 文件拖拽处理 - 使用 Label 原生触发文件选择，无需 JS 点击模拟
function setupDragDrop() {
    // 防止默认拖拽行为
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // 拖拽视觉反馈
    ['dragenter', 'dragover'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, unhighlight, false);
    });

    // 处理文件拖拽放下
    elements.dropZone.addEventListener('drop', handleDrop, false);

    // 选择文件后的回调
    elements.audioInput.addEventListener('change', handleFileSelect);

    // 点击兜底：部分浏览器里 label 包着隐藏 input 时不会打开选择框，
    // 这里统一用 JS 打开（先 preventDefault 阻止 label 的默认转发，避免弹两次）。
    elements.dropZone.addEventListener('click', (e) => {
        // input.click() 触发的事件也会冒泡到这里，必须放行，否则会无限递归
        if (e.target === elements.audioInput) return;
        e.preventDefault();
        elements.audioInput.click();
    });

    // 移除按钮
    elements.removeFile.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        clearFile();
    });
    elements.removeFile.addEventListener('touchend', (e) => {
        e.stopPropagation();
        e.preventDefault();
        clearFile();
    }, { passive: false });

    // 文件信息区域点击/触摸重新选择 - 直接触发隐藏 input 的 click
    // 注意：fileInfo 在 Label 外部，需要 JS 触发
    const reSelectFile = (e) => {
        if (e.target.closest('#removeFile')) return;
        elements.audioInput.click();
    };
    elements.fileInfo.addEventListener('click', reSelectFile);
    elements.fileInfo.addEventListener('touchend', (e) => {
        e.preventDefault();
        reSelectFile(e);
    }, { passive: false });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    elements.dropZone.classList.add('drag-over');
}

function unhighlight() {
    elements.dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    if (!dt) return;
    const files = dt.files;
    if (files && files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
        handleFile(files[0]);
    }
}

// Windows 上不少 .flac / .m4a 的 file.type 是空字符串，
// 只按 MIME 判断会导致“选完文件被静默拒绝”，这里补上扩展名兜底。
const AUDIO_EXTENSIONS = ['mp3', 'wav', 'flac', 'ogg', 'oga', 'm4a', 'aac', 'opus', 'webm', 'wma', 'aiff', 'aif'];

function isAudioFile(file) {
    const type = (file.type || '').toLowerCase();
    if (type.startsWith('audio/')) return true;
    // 有些系统会把 m4a / webm 音频识别成 video/*
    if (type === 'video/mp4' || type === 'video/webm' || type === 'video/ogg') return true;
    const name = (file.name || '').toLowerCase();
    return AUDIO_EXTENSIONS.some(ext => name.endsWith('.' + ext));
}

function handleFile(file) {
    if (!isAudioFile(file)) {
        showToast('请选择音频文件', 'error');
        return;
    }

    // 检查文件大小 (限制 100MB)
    if (file.size > 100 * 1024 * 1024) {
        showToast('文件过大，请选择小于 100MB 的文件', 'error');
        return;
    }

    state.audioFile = file;
    state.audioBuffer = null;
    updateFileInfo(file);
    loadAudioFile(file);
}

function updateFileInfo(file) {
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    elements.dropZone.classList.add('has-file');
    elements.fileInfo.hidden = false;
    elements.processBtn.disabled = false;
}

function clearFile() {
    state.audioFile = null;
    state.audioBuffer = null;
    elements.dropZone.classList.remove('has-file');
    elements.fileInfo.hidden = true;
    elements.audioInput.value = '';
    elements.processBtn.disabled = true;
    hideResults();
}

// 复用同一个 AudioContext（每次 new 会快速耗尽浏览器上限）
let sharedAudioContext = null;

function getAudioContext() {
    if (!sharedAudioContext) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) {
            throw new Error('当前浏览器不支持 Web Audio API');
        }
        sharedAudioContext = new Ctx();
    }
    if (sharedAudioContext.state === 'suspended') {
        sharedAudioContext.resume().catch(() => {});
    }
    return sharedAudioContext;
}

// 加载音频文件为 AudioBuffer
// 旧实现依赖 <audio> 的 canplaythrough 事件，该事件在某些格式/浏览器上不会触发，
// 于是“选完文件”后既不报错也没有任何反馈。这里直接解码，失败时给出明确提示。
async function loadAudioFile(file) {
    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = getAudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        state.audioBuffer = audioBuffer;

        const channelText = audioBuffer.numberOfChannels === 1 ? '单声道' : `${audioBuffer.numberOfChannels} 声道`;
        elements.fileSize.textContent = `${formatFileSize(file.size)} · ${channelText} · ${formatTime(audioBuffer.duration)}`;
        showToast('音频加载完成，可以开始分离', 'success');
    } catch (error) {
        console.error('音频解码失败:', error);
        state.audioBuffer = null;
        elements.processBtn.disabled = true;
        showToast('音频解码失败，请换一个文件试试（建议 MP3 / WAV）', 'error');
    }
}

// 模型选择处理
function setupModelSelection() {
    elements.modelRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.checked) {
                state.selectedModel = e.target.value;
                // 切换模型只需要丢弃旧会话，真正下载推迟到点击“开始分离”时
                state.modelLoaded = false;
                state.session = null;
                // 换模型就忽略之前手动选过的本地文件，避免形状不匹配
                state.localModelData = null;
                if (elements.localModelName) elements.localModelName.textContent = '';
                elements.modelProgress.hidden = true;
            }
        });
    });

    // 本地模型：镜像被墙时可以直接加载已经下载到磁盘的 .onnx
    if (elements.localModelInput) {
        elements.localModelInput.addEventListener('change', async (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;

            if (elements.localModelName) elements.localModelName.textContent = `正在读取 ${file.name} ...`;
            try {
                state.localModelData = await file.arrayBuffer();
                state.session = null;
                state.modelLoaded = false;
                elements.modelProgress.hidden = true;
                if (elements.localModelName) {
                    elements.localModelName.textContent =
                        `已选择本地模型：${file.name}（${formatFileSize(file.size)}）`;
                }
                showToast('本地模型已就绪，点击“开始分离”使用', 'success');
            } catch (error) {
                console.error('读取本地模型失败:', error);
                state.localModelData = null;
                if (elements.localModelName) elements.localModelName.textContent = '';
                showToast('读取本地模型失败：' + error.message, 'error');
            }
        });
    }
}

let modelProgressTimer = null;

function startModelProgress(config) {
    elements.modelProgress.hidden = false;
    elements.modelProgressFill.style.width = '0%';
    elements.modelProgressFill.style.background = '';
    elements.modelProgressText.textContent =
        `正在准备 ${config.name} 模型（约 ${config.size}）...`;
}

function updateModelProgress(received, total) {
    const receivedMB = (received / 1024 / 1024).toFixed(1);
    let text;

    if (total > 0) {
        const percent = Math.min((received / total) * 100, 100);
        text = `正在下载模型 ${receivedMB}MB / ${(total / 1024 / 1024).toFixed(0)}MB (${percent.toFixed(0)}%)`;
        elements.modelProgressFill.style.width = `${percent.toFixed(1)}%`;
        elements.modelProgressText.textContent = text;

        // 同步到下方的转换进度条，让"点完按钮"立刻有整体反馈
        if (elements.processProgress && !elements.processProgress.hidden) {
            updateProcessProgress(percent, '下载模型', text);
        }
    } else {
        text = `正在下载模型 ${receivedMB}MB ...`;
        elements.modelProgressText.textContent = text;

        if (elements.processProgress && !elements.processProgress.hidden) {
            setProcessIndeterminate('下载模型', text);
        }
    }
}

function stopModelProgress(success) {
    clearInterval(modelProgressTimer);
    modelProgressTimer = null;

    if (success) {
        elements.modelProgressFill.style.width = '100%';
        elements.modelProgressText.textContent = '模型加载完成！';
        setTimeout(() => {
            elements.modelProgress.hidden = true;
        }, 1200);
    } else {
        elements.modelProgressFill.style.background = 'var(--color-error)';
    }
}

// 带真实进度的模型下载（316MB 的模型如果不显示进度，用户会以为卡死了）
async function fetchModelWithProgress(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const total = Number(response.headers.get('content-length')) || 0;

    if (!response.body || typeof response.body.getReader !== 'function') {
        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
    }

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        onProgress(received, total);
    }

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}

// 懒加载模型：页面打开时不下载，点击“开始分离”才加载
async function ensureModel() {
    if (state.session) return true;

    const config = MODEL_CONFIGS[state.selectedModel];
    if (!config) {
        showToast('未知的模型配置', 'error');
        return false;
    }

    startModelProgress(config);
    setProcessIndeterminate('准备引擎', '正在加载 ONNX Runtime 推理引擎（首次约 10MB，之后走缓存）...');

    try {
        const runtime = await loadOrt();
        setProcessIndeterminate('准备模型', '正在准备模型文件...');

        // GitHub Pages 没有开启 COOP/COEP，SharedArrayBuffer 不可用，只能单线程
        try {
            runtime.env.wasm.numThreads = 1;
        } catch (e) {
            // 某些版本该属性只读，忽略即可
        }

        let modelData = state.localModelData;
        if (!modelData) {
            const errors = [];
            for (const url of config.urls) {
                try {
                    elements.modelProgressText.textContent = `正在连接模型服务器...`;
                    modelData = await fetchModelWithProgress(url, updateModelProgress);
                    break;
                } catch (error) {
                    console.warn('模型下载失败:', url, error);
                    errors.push(`${url} → ${error.message}`);
                }
            }
            if (!modelData) {
                throw new Error(
                    '模型文件下载失败，可能被网络或镜像拦截。\n' +
                    '可以把模型下载到本地后，用下面的「选择本地模型文件」加载。\n' +
                    errors.join('\n')
                );
            }
        }

        elements.modelProgressText.textContent = '正在初始化推理会话（首次较慢，请耐心等待）...';
        state.session = await runtime.InferenceSession.create(modelData, {
            executionProviders: ['wasm'],
            // 用 basic 就够了：htdemucs 有 ~2.5 万个节点，extended/all 级别的图优化
            // 在浏览器里既慢又吃内存，收益很小
            graphOptimizationLevel: 'basic'
        });

        // 释放 300MB 的模型字节，避免和 wasm 内存叠加
        modelData = null;
        state.localModelData = null;

        // 不同导出版本的输入输出名可能不同，这里按实际会话校正
        state.inputName = state.session.inputNames.includes(config.inputName)
            ? config.inputName
            : state.session.inputNames[0];
        state.outputName = state.session.outputNames.includes(config.outputName)
            ? config.outputName
            : state.session.outputNames[0];

        state.modelLoaded = true;
        stopModelProgress(true);
        showToast(`${config.name} 模型加载成功`, 'success');
        return true;
    } catch (error) {
        console.error('模型加载失败:', error);
        state.session = null;
        state.modelLoaded = false;
        stopModelProgress(false);

        const raw = String((error && error.message) || error);
        // 浏览器里最常见的两种失败：WASM 内存不够（abort）、或者模型文件取不到
        const isMemoryIssue = /abort|memory|allocat/i.test(raw);
        elements.modelProgressText.textContent = isMemoryIssue
            ? '模型加载失败：浏览器内存不足（也可能是该模型不被当前浏览器支持）。请关闭其他标签页/程序后重试，或改用「轻量版」/本地模型文件'
            : '模型加载失败，请检查网络后重试（或使用本地模型文件）';
        showToast('模型加载失败：' + raw, 'error');
        return false;
    }
}

// ---------- 转换进度条 ----------
function showProcessProgress(stage, text) {
    elements.processProgress.hidden = false;
    elements.processBar.classList.remove('is-indeterminate');
    elements.processProgressFill.classList.add('is-busy');
    elements.processProgressFill.style.background = '';
    elements.processProgressFill.style.width = '0%';
    elements.processPercent.textContent = '0%';
    elements.processStage.textContent = stage;
    elements.processProgressText.textContent = text || '';
    elements.processedTime.textContent = '0:00';
    elements.totalTime.textContent = '--:--';
}

// 进度未知的步骤（例如首次准备模型）用滑动条表示"还在干活"
function setProcessIndeterminate(stage, text) {
    elements.processProgress.hidden = false;
    elements.processBar.classList.add('is-indeterminate');
    elements.processProgressFill.classList.remove('is-busy');
    elements.processPercent.textContent = '--';
    elements.processStage.textContent = stage;
    elements.processProgressText.textContent = text || '';
}

function updateProcessProgress(percent, stage, text) {
    const value = Math.max(0, Math.min(100, percent || 0));
    elements.processBar.classList.remove('is-indeterminate');
    elements.processProgressFill.classList.add('is-busy');
    elements.processProgressFill.style.width = `${value.toFixed(1)}%`;
    elements.processPercent.textContent = `${Math.round(value)}%`;
    if (stage) elements.processStage.textContent = stage;
    if (text) elements.processProgressText.textContent = text;
}

function updateProcessStats(elapsedSeconds, remainingSeconds) {
    elements.processedTime.textContent = formatTime(elapsedSeconds);
    elements.totalTime.textContent = remainingSeconds == null ? '--:--' : formatTime(remainingSeconds);
}

function finishProcessProgress(text) {
    elements.processBar.classList.remove('is-indeterminate');
    elements.processProgressFill.classList.remove('is-busy');
    elements.processProgressFill.style.background = '';
    elements.processProgressFill.style.width = '100%';
    elements.processPercent.textContent = '100%';
    elements.processStage.textContent = '转换完成';
    if (text) elements.processProgressText.textContent = text;
}

function hideProcessProgress() {
    elements.processProgress.hidden = true;
    elements.processBar.classList.remove('is-indeterminate');
    elements.processProgressFill.classList.remove('is-busy');
}

// 音频处理主流程
async function processAudio() {
    if (state.isProcessing) return;

    if (!state.audioBuffer) {
        showToast('请先选择音频文件', 'error');
        return;
    }

    // 浏览器端单线程推理很慢，长音频提前提醒（不阻止，用户可以继续等）
    if (state.audioBuffer.duration > 120) {
        showToast('音频超过 2 分钟，浏览器处理会比较慢，建议先用短片段试效果', 'info');
    }

    // 首次运行需要先下载模型，期间禁用按钮避免重复点击
    if (!state.session) {
        state.isProcessing = true;
        elements.processBtn.disabled = true;
        setProcessIndeterminate('准备模型', '首次使用需要下载模型文件，请稍候...');

        const ready = await ensureModel();

        state.isProcessing = false;
        if (!ready) {
            elements.processBtn.disabled = false;
            hideProcessProgress();
            return;
        }
    }

    state.isProcessing = true;
    elements.processBtn.disabled = true;
    elements.processBtn.querySelector('.btn-text').hidden = true;
    elements.processBtn.querySelector('.btn-loading').hidden = false;
    showProcessProgress('准备转换', '正在解码音频数据...');
    hideResults();

    try {
        const config = MODEL_CONFIGS[state.selectedModel];
        const results = await separateAudio(state.audioBuffer, config);
        displayResults(results);
        // AudioBuffer 里的数据已经编码成 WAV Blob，这里释放掉，给长音频省内存
        state.results = {};
        finishProcessProgress('已生成人声、伴奏、鼓、贝斯、其他');
        await new Promise((resolve) => setTimeout(resolve, 700));
        showToast('分离完成！', 'success');
    } catch (error) {
        console.error('处理失败:', error);
        elements.processStage.textContent = '转换失败';
        elements.processProgressText.textContent = error.message || String(error);
        elements.processBar.classList.remove('is-indeterminate');
        elements.processProgressFill.classList.remove('is-busy');
        elements.processProgressFill.style.background = 'var(--color-error)';
        showToast('处理失败: ' + error.message, 'error');
    } finally {
        state.isProcessing = false;
        elements.processBtn.disabled = false;
        elements.processBtn.querySelector('.btn-text').hidden = false;
        elements.processBtn.querySelector('.btn-loading').hidden = true;
        // 失败时保留错误信息，几秒后再收起进度条
        setTimeout(hideProcessProgress, 1500);
    }
}

// 音频分离核心算法
// 与 htdemucs.onnx 的真实契约对齐：
//   输入 mix   [1, 2, 343980]  (7.8s @ 44.1kHz 立体声)
//   输出 stems [1, 4, 2, 343980]，第 2 维顺序固定为 [drums, bass, other, vocals]
async function separateAudio(audioBuffer, config) {
    const { sampleRate, segmentLength, overlap, sources } = config;

    if (audioBuffer.duration > 600) {
        throw new Error('音频超过 10 分钟，浏览器内存不足以处理，请先裁剪');
    }

    // 重采样到模型要求的采样率 (如果需要)
    let inputBuffer = audioBuffer;
    if (Math.abs(audioBuffer.sampleRate - sampleRate) > 1) {
        inputBuffer = await resampleAudio(audioBuffer, sampleRate);
    }

    const totalLength = inputBuffer.length;
    const inputChannels = inputBuffer.numberOfChannels;
    const numChannels = 2; // 模型固定双声道，单声道文件复制成双声道
    const stemCount = sources.length;

    const hopSize = Math.max(1, Math.floor(segmentLength * (1 - overlap)));
    const numSegments = totalLength <= segmentLength
        ? 1
        : Math.ceil((totalLength - segmentLength) / hopSize) + 1;

    // 输出缓冲区（声道优先布局：ch * length + i）
    const outputBuffers = {};
    sources.forEach((source) => {
        outputBuffers[source] = new Float32Array(numChannels * totalLength);
    });

    // Hann 窗 (用于重叠相加)
    const window = new Float32Array(segmentLength);
    for (let i = 0; i < segmentLength; i++) {
        window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (segmentLength - 1)));
    }

    const startTime = Date.now();

    showProcessProgress('转换中', `共 ${numSegments} 段，正在处理第 1 段...`);

    // 段落之间才更新百分比；段内靠"已用时间"+斜纹动画表示仍在工作
    const tickTimer = setInterval(() => {
        elements.processedTime.textContent = formatTime((Date.now() - startTime) / 1000);
    }, 1000);

    try {
        for (let seg = 0; seg < numSegments; seg++) {
            updateProcessProgress(
                (seg / numSegments) * 100,
                '转换中',
                `正在处理第 ${seg + 1}/${numSegments} 段...`
            );

            const start = seg * hopSize;
            const end = Math.min(start + segmentLength, totalLength);
            const actualLength = end - start;

            // 组装输入片段 [1, 2, segmentLength]，末尾不足部分补零
            const segmentInput = new Float32Array(numChannels * segmentLength);
            for (let ch = 0; ch < numChannels; ch++) {
                const channelData = inputBuffer.getChannelData(Math.min(ch, inputChannels - 1));
                const target = ch * segmentLength;
                for (let i = 0; i < actualLength; i++) {
                    segmentInput[target + i] = channelData[start + i] * window[i];
                }
            }

            const tensor = new ort.Tensor('float32', segmentInput, [1, numChannels, segmentLength]);
            const feeds = {};
            feeds[state.inputName] = tensor;

            const outputs = await state.session.run(feeds);
            const stemTensor = outputs[state.outputName];

            if (!stemTensor || !stemTensor.data) {
                throw new Error('模型输出与预期不符，实际输出: ' + (Object.keys(outputs).join(', ') || '无'));
            }

            const dims = stemTensor.dims || [];
            if (dims.length !== 4 || dims[1] !== stemCount || dims[2] !== numChannels || dims[3] !== segmentLength) {
                throw new Error(
                    `模型输出形状异常: [${dims.join(', ')}]，期望 [1, ${stemCount}, ${numChannels}, ${segmentLength}]`
                );
            }

            // [1, 4, 2, T] -> 逐路重叠相加
            const data = stemTensor.data;
            for (let s = 0; s < stemCount; s++) {
                const outChannel = outputBuffers[sources[s]];
                for (let ch = 0; ch < numChannels; ch++) {
                    const base = ((s * numChannels) + ch) * segmentLength;
                    const outBase = ch * totalLength + start;
                    for (let i = 0; i < actualLength; i++) {
                        outChannel[outBase + i] += data[base + i] * window[i];
                    }
                }
            }

            // 更新进度：百分比 + 已用/剩余时间
            const elapsed = (Date.now() - startTime) / 1000;
            const perSegment = elapsed / (seg + 1);
            const remaining = perSegment * (numSegments - seg - 1);
            const percent = ((seg + 1) / numSegments) * 100;

            updateProcessProgress(
                percent,
                '转换中',
                `已完成 ${seg + 1}/${numSegments} 段 · 平均每段 ${perSegment.toFixed(1)}s`
            );
            updateProcessStats(elapsed, seg + 1 < numSegments ? remaining : 0);

            // 让出主线程，避免页面假死
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
    } finally {
        clearInterval(tickTimer);
    }

    // 重叠区域的归一化权重
    const overlapAddBuffer = new Float32Array(totalLength);
    for (let seg = 0; seg < numSegments; seg++) {
        const start = seg * hopSize;
        const end = Math.min(start + segmentLength, totalLength);
        for (let i = start; i < end; i++) {
            overlapAddBuffer[i] += window[i - start] * window[i - start];
        }
    }

    const ctx = getAudioContext();
    const resultBuffers = {};

    // 就地归一化，避免再复制一份 4 轨的 Float32Array（长音频很吃内存）
    sources.forEach((source) => {
        const data = outputBuffers[source];
        for (let ch = 0; ch < numChannels; ch++) {
            const offset = ch * totalLength;
            for (let i = 0; i < totalLength; i++) {
                const weight = overlapAddBuffer[i] > 0 ? overlapAddBuffer[i] : 1;
                data[offset + i] /= weight;
            }
        }
    });

    // 伴奏 = 原曲 - 人声（模型只输出 4 个源，这里补齐用户最关心的“伴奏”）
    if (outputBuffers.vocals) {
        const vocals = outputBuffers.vocals;
        const accompaniment = new Float32Array(numChannels * totalLength);
        for (let ch = 0; ch < numChannels; ch++) {
            const offset = ch * totalLength;
            const channelData = inputBuffer.getChannelData(Math.min(ch, inputChannels - 1));
            for (let i = 0; i < totalLength; i++) {
                accompaniment[offset + i] = channelData[i] - vocals[offset + i];
            }
        }
        outputBuffers.accompaniment = accompaniment;
    }

    Object.keys(outputBuffers).forEach((source) => {
        const data = outputBuffers[source];
        const buffer = ctx.createBuffer(numChannels, totalLength, sampleRate);
        for (let ch = 0; ch < numChannels; ch++) {
            buffer.copyToChannel(data.subarray(ch * totalLength, (ch + 1) * totalLength), ch);
        }
        resultBuffers[source] = buffer;
    });

    return resultBuffers;
}

// 音频重采样
function resampleAudio(audioBuffer, targetSampleRate) {
    return new Promise((resolve, reject) => {
        const offlineContext = new OfflineAudioContext(
            audioBuffer.numberOfChannels,
            Math.round(audioBuffer.length * targetSampleRate / audioBuffer.sampleRate),
            targetSampleRate
        );
        const bufferSource = offlineContext.createBufferSource();
        bufferSource.buffer = audioBuffer;
        bufferSource.connect(offlineContext.destination);
        bufferSource.start(0);
        offlineContext.startRendering().then(resolve).catch(reject);
    });
}

// 音源 -> 结果卡片 id 前缀（注意人声卡片的 id 是 vocal，不是 vocals）
const SOURCE_UI = {
    vocals: 'vocal',
    accompaniment: 'accompaniment',
    drums: 'drums',
    bass: 'bass',
    other: 'other'
};

// 显示结果
function displayResults(results) {
    elements.resultsSection.hidden = false;

    Object.keys(SOURCE_UI).forEach((source) => {
        const prefix = SOURCE_UI[source];
        const card = document.getElementById(`${prefix}Card`);
        const audio = document.getElementById(`${prefix}Audio`);
        const download = document.getElementById(`${prefix}Download`);
        const duration = document.getElementById(`${prefix}Duration`);
        const buffer = results[source];

        if (!card) return;

        if (!buffer) {
            card.hidden = true;
            return;
        }

        card.hidden = false;
        const blob = audioBufferToWav(buffer);
        const url = URL.createObjectURL(blob);
        audio.src = url;
        audio.onloadedmetadata = () => {
            duration.textContent = formatTime(audio.duration);
        };
        download.href = url;
        download.download = `${source}_${state.audioFile.name.replace(/\.[^/.]+$/, '')}.wav`;
    });

    // 滚动到结果区域
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// AudioBuffer 转 WAV Blob
function audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length * numChannels * 2 + 44;
    const buffer = new ArrayBuffer(length);
    const view = new DataView(buffer);

    // WAV 头部
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, length - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length - 44, true);

    // 写入音频数据
    let offset = 44;
    for (let i = 0; i < audioBuffer.length; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
            const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(ch)[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function hideResults() {
    elements.resultsSection.hidden = true;
    // 释放之前的 Blob URL
    [elements.vocalAudio, elements.accompanimentAudio, elements.drumsAudio, elements.bassAudio, elements.otherAudio].forEach(audio => {
        if (audio.src && audio.src.startsWith('blob:')) {
            URL.revokeObjectURL(audio.src);
        }
        audio.src = '';
    });
}

// 处理另一个文件
function processAnother() {
    clearFile();
    hideResults();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 初始化应用
async function init() {
    initElements();
    setupDragDrop();
    setupModelSelection();

    elements.processBtn.addEventListener('click', processAudio);
    elements.processAnother.addEventListener('click', processAnother);

    // 不再在页面打开时自动下载 100MB+ 的模型：
    // 既拖慢首屏，也会让用户以为“页面卡住了”。模型改为点击“开始分离”时懒加载。

    // 通知 index.html 里的兜底脚本：应用已正常启动
    window.__appReady = true;
    if (typeof window.clearBootError === 'function') {
        window.clearBootError();
    }
}

// 启动（模块脚本在解析后执行，此时 DOMContentLoaded 可能已经过去）
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}