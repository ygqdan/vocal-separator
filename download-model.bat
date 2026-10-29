@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo   HTDemucs 模型下载器
echo   ========================================
echo.

set "VARIANT=fp32"
if /i "%~1"=="fp16" set "VARIANT=fp16"
if /i "%~1"=="small" set "VARIANT=fp16"

if /i "%VARIANT%"=="fp16" (
    set "FILE=htdemucs_fp16weights.onnx"
    set "EXPECT=165612636"
    set "SIZEHINT=约 166MB"
) else (
    set "FILE=htdemucs.onnx"
    set "EXPECT=316446953"
    set "SIZEHINT=约 316MB"
)

if not exist "models" mkdir "models"

REM 先写到 .part，校验大小通过后才改名，避免页面读到半个模型
set "PART=models\%FILE%.part"
set "FINAL=models\%FILE%"

echo   目标文件: %FINAL%
echo   文件大小: %SIZEHINT%
echo.

if not exist "%FINAL%" goto :download
for %%A in ("%FINAL%") do set "EXISTING=%%~zA"
if not "%EXISTING%"=="%EXPECT%" goto :download

echo   模型已存在且大小正确，无需下载。
goto :end

:download
if exist "%PART%" goto :fetch
if not exist "%FINAL%" goto :fetch

echo   发现未下载完的文件，转为断点续传...
move /y "%FINAL%" "%PART%" >nul

:fetch
set "URL1=https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/%FILE%"
set "URL2=https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/%FILE%"

echo   [1/2] 从 hf-mirror.com 下载（支持续传，可随时 Ctrl+C 后重跑）...
curl.exe -L --retry 3 --retry-delay 2 -C - -o "%PART%" "%URL1%"
if not errorlevel 1 goto :verify

echo.
echo   hf-mirror 失败，改用 huggingface.co 重试...
curl.exe -L --retry 3 --retry-delay 2 -C - -o "%PART%" "%URL2%"
if errorlevel 1 goto :failed

:verify
for %%A in ("%PART%") do set "BYTES=%%~zA"
if "%BYTES%"=="%EXPECT%" goto :ok

echo.
echo   [警告] 文件大小不符: 实际 %BYTES% 字节, 期望 %EXPECT% 字节。
echo          网络可能中断了，重新运行本脚本会继续续传。
goto :end

:ok
move /y "%PART%" "%FINAL%" >nul
echo.
echo   [完成] %FINAL%  (%BYTES% 字节)
echo.
echo   刷新页面并点击 "开始分离" 即可，页面会优先使用本地模型文件。
goto :end

:failed
echo.
echo   [错误] 下载失败，请检查网络 / 代理设置，或手动下载:
echo     %URL1%
echo   并保存为: %FINAL%
goto :end

:end
echo.
pause
