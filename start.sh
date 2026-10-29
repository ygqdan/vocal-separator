#!/bin/bash

# 音乐人声分离器 - 本地启动脚本

echo "🎵 音乐人声分离器 - 本地服务器启动器"
echo "========================================"

# 检查项目目录
if [ ! -f "index.html" ]; then
    echo "❌ 错误: 请在 vocal-separator 目录下运行此脚本"
    exit 1
fi

# 尝试不同的静态服务器
PORT=8000

echo "🔍 检测可用的静态服务器..."

if command -v python3 &> /dev/null; then
    echo "✅ 发现 Python 3"
    echo "🚀 启动服务器: http://localhost:$PORT"
    echo "📝 按 Ctrl+C 停止服务器"
    echo ""
    python3 -m http.server $PORT
elif command -v python &> /dev/null; then
    echo "✅ 发现 Python"
    echo "🚀 启动服务器: http://localhost:$PORT"
    echo "📝 按 Ctrl+C 停止服务器"
    echo ""
    python -m http.server $PORT
elif command -v npx &> /dev/null; then
    echo "✅ 发现 npx (Node.js)"
    echo "🚀 启动服务器: http://localhost:$PORT"
    echo "📝 按 Ctrl+C 停止服务器"
    echo ""
    npx serve . -p $PORT
elif command -v php &> /dev/null; then
    echo "✅ 发现 PHP"
    echo "🚀 启动服务器: http://localhost:$PORT"
    echo "📝 按 Ctrl+C 停止服务器"
    echo ""
    php -S localhost:$PORT
else
    echo "❌ 未找到可用的静态服务器"
    echo "请安装以下任一工具:"
    echo "  - Python 3: python3 -m http.server 8000"
    echo "  - Node.js: npx serve ."
    echo "  - PHP: php -S localhost:8000"
    exit 1
fi