#!/usr/bin/env bash
# 启动本地 Demucs 服务（桌面加速模式）
# 注意：torch 在过新的 Python（如 3.14）上可能装不上/导入失败，这里建独立 .venv。
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 未检测到 python3，请先安装 Python 3.11~3.13"
    exit 1
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "使用 Python: $(command -v python3) ($PYVER)"
case "$PYVER" in
    3.14|3.15)
        echo "[警告] 当前 Python $PYVER 可能不被 torch 支持，建议改用 3.11~3.13。"
        echo "仍继续尝试安装，若 torch 导入失败请换 Python 版本。"
        ;;
esac

if [ ! -x .venv/bin/python ]; then
    echo "正在创建虚拟环境 .venv（仅首次）..."
    python3 -m venv .venv
fi
VPY=".venv/bin/python"

echo "[1/2] 检查依赖（首次会自动 pip 安装，torch 较大）..."
if ! "$VPY" -c "import flask, flask_cors, demucs, torch" >/dev/null 2>&1; then
    "$VPY" -m pip install -r requirements.txt
fi

echo "[2/2] 启动本地 Demucs 服务（默认 http://127.0.0.1:8001）..."
echo "保持此终端打开；Ctrl+C 停止服务。"
"$VPY" server.py "$@"
