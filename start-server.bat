@echo off
chcp 65001 >nul
REM 启动本地 Demucs 服务（桌面加速模式）
REM 网页会自动探测 http://127.0.0.1:8001，连上就切到本地模式。
REM
REM 注意：torch 在过新的 Python（如 3.14）上可能装不上/导入失败，
REM 所以这里优先找 3.11~3.13 的 Python，并在项目内建 .venv 虚拟环境。

cd /d %~dp0

REM ---- 1) 找一个 torch 支持的 Python（3.11 / 3.12 / 3.13）----
set "PYEXE="
py -3.12 --version >nul 2>&1 && set "PYEXE=py -3.12"
if not defined PYEXE py -3.11 --version >nul 2>&1 && set "PYEXE=py -3.11"
if not defined PYEXE py -3.13 --version >nul 2>&1 && set "PYEXE=py -3.13"
if not defined PYEXE (
    echo [错误] 未找到 Python 3.11 / 3.12 / 3.13。
    echo 当前 torch 尚不支持 Python 3.14，请先安装其中一个版本（推荐 3.12）。
    pause
    exit /b 1
)
echo 使用 Python: %PYEXE%

REM ---- 2) 准备项目内虚拟环境 .venv ----
if not exist .venv\Scripts\python.exe (
    echo 正在创建虚拟环境 .venv（仅首次）...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

echo [1/2] 检查依赖（首次会自动 pip 安装，torch 较大，可能耗时几分钟）...
"%VPY%" -c "import flask, flask_cors, demucs, torch" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖：flask flask-cors demucs ...
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动执行：
        echo     "%VPY%" -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [2/2] 启动本地 Demucs 服务（默认 http://127.0.0.1:8001）...
echo 保持此窗口打开；关闭窗口即停止服务。

REM Demucs 模型要从 Hugging Face 下载，国内用镜像更稳（与网页端一致）
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"

REM torch.hub / demucs 默认缓存到 ~/.cache/torch，某些环境会“拒绝访问”，
REM 改到项目内 .cache，自包含且保证可写。
if not defined TORCH_HOME set "TORCH_HOME=%~dp0.cache"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%" >nul 2>&1

"%VPY%" server.py %*
pause
