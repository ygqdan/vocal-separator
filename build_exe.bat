@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [错误] 未检测到 .venv，请先运行 start-server.bat 创建 Python 环境。
    pause
    exit /b 1
)

echo [1/2] 安装/更新 PyInstaller...
.venv\Scripts\pip install -U pyinstaller

echo [2/2] 开始打包（onedir，含模型缓存，耗时较长，请耐心等待）...
.venv\Scripts\pyinstaller --noconfirm --onedir --name vocal-separator --add-data "index.html;." --add-data "app.js;." --add-data "styles.css;." --add-data ".cache;./.cache" --hidden-import torch --hidden-import demucs --hidden-import faster_whisper --hidden-import flask --hidden-import flask_cors --hidden-import ctranslate2 --hidden-import av --hidden-import onnxruntime --hidden-import huggingface_hub --hidden-import tokenizers --hidden-import safetensors --hidden-import soundfile --hidden-import numpy --collect-all torch --collect-all ctranslate2 --collect-all av --exclude-module torchaudio server.py

echo.
echo 打包完成！产物在 dist\vocal-separator\ 目录下。
echo 双击 dist\vocal-separator\vocal-separator.exe 即可启动（会自动打开浏览器）。
echo 如需单个安装包，可把 dist\vocal-separator\ 目录用 NSIS/Inno Setup 封装。
endlocal
