@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

title 音乐人声分离器 - 本地服务器

echo.
echo  音乐人声分离器 - 本地服务器启动器
echo  ========================================
echo.

if not exist "index.html" (
    echo [错误] 请在 vocal-separator 目录下运行此脚本
    echo        当前目录: %CD%
    echo.
    pause
    exit /b 1
)

set "PORT=8000"

REM 可选: 传入第一个参数指定端口, 例如  start.bat 8080
if not "%~1"=="" set "PORT=%~1"

echo [1/2] 检测可用的静态服务器...
echo.

set "SERVER="
set "OPEN_BROWSER=0"

REM ---- Python 3 (py 启动器) ----
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "SERVER=py -3 -m http.server %PORT%"
        set "OPEN_BROWSER=1"
        goto :found
    )
)

REM ---- Python 3 (python 命令, 排除应用商店的假 python) ----
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "SERVER=python -m http.server %PORT%"
        set "OPEN_BROWSER=1"
        goto :found
    )
)

REM ---- Node.js (npx serve) ----
where npx >nul 2>nul
if not errorlevel 1 (
    set "SERVER=npx --yes serve . -p %PORT%"
    goto :found
)

REM ---- PHP ----
where php >nul 2>nul
if not errorlevel 1 (
    set "SERVER=php -S localhost:%PORT%"
    set "OPEN_BROWSER=1"
    goto :found
)

echo [错误] 未找到可用的静态服务器
echo.
echo 请安装以下任一工具后重新运行本脚本:
echo   - Python 3   https://www.python.org/downloads/
echo   - Node.js    https://nodejs.org/
echo   - PHP        https://windows.php.net/
echo.
echo 也可以手动启动, 例如:
echo   python -m http.server %PORT%
echo.
pause
exit /b 1

:found
echo [2/2] 已选择: %SERVER%
echo.
echo   访问地址: http://localhost:%PORT%
echo   停止服务: 按 Ctrl+C
echo.

REM 设置 NO_OPEN=1 可禁止自动打开浏览器
if "%NO_OPEN%"=="1" set "OPEN_BROWSER=0"

if "%OPEN_BROWSER%"=="1" (
    start "" "http://localhost:%PORT%"
)

%SERVER%

echo.
echo 服务器已停止
pause
