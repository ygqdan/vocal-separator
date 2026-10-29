# 🎵 音乐人声分离器

一个纯前端的音乐人声分离网页应用，基于 ONNX.js 在浏览器中运行 AI 模型，无需服务器部署。

## ✨ 功能特性

- **完全在浏览器运行** - 无需上传文件到服务器，保护隐私
- **多模型支持** - HTDemucs (高质量4源)、MDX-Net (轻量2源)、UMX-L (经典开源)
- **拖拽上传** - 支持 MP3, WAV, FLAC, OGG, M4A 等常见音频格式
- **实时进度显示** - 模型下载、音频处理进度可视化
- **多轨道输出** - 人声、伴奏、鼓、贝斯、其他乐器分离下载
- **响应式设计** - 适配桌面端和移动端
- **深色模式支持** - 自动跟随系统主题

## 🚀 快速开始

### 在线体验

直接打开 `index.html` 即可使用，或部署到任意静态托管平台：
- GitHub Pages
- Vercel
- Netlify
- Cloudflare Pages
- 任何支持静态文件的 Web 服务器

### 本地运行

```bash
# 进入项目目录
cd vocal-separator

# 使用任意静态服务器
# Python 3
python -m http.server 8000

# Node.js (需要安装 serve)
npx serve .

# PHP
php -S localhost:8000
```

然后在浏览器打开 `http://localhost:8000`

## 📦 模型说明

| 模型 | 参数量 | 分离轨道 | 适用场景 | 首次下载 |
|------|--------|----------|----------|----------|
| **HTDemucs** | ~120MB | 人声、鼓、贝斯、其他 | 专业级分离，质量最佳 | ~120MB |
| **MDX-Net** | ~45MB | 人声、伴奏 | 快速分离，仅需人声/伴奏 | ~45MB |
| **UMX-L** | ~80MB | 人声、鼓、贝斯、其他 | 兼容性好，经典开源模型 | ~80MB |

> ⚠️ **首次使用需下载模型文件**，建议在 WiFi 环境下使用。模型会缓存到浏览器，后续使用无需重复下载。

## 🛠 技术栈

- **ONNX Runtime Web** - 浏览器端 ONNX 模型推理
- **Web Audio API** - 音频解码、重采样、处理
- **原生 JavaScript (ES Modules)** - 无框架依赖，轻量快速
- **CSS 变量 + 现代 CSS** - 设计系统，支持深色模式

## 📁 项目结构

```
vocal-separator/
├── index.html      # 主页面
├── styles.css      # 样式文件
├── app.js          # 主逻辑
└── README.md       # 说明文档
```

## 🔧 核心原理

1. **音频加载** - 使用 Web Audio API 解码音频文件为 AudioBuffer
2. **重采样** - 统一重采样到模型要求的 44.1kHz
3. **分段推理** - 将长音频切分为重叠片段，逐段送入 ONNX 模型
4. **重叠相加 (OLA)** - 使用 Hann 窗函数平滑拼接分段输出
5. **导出下载** - 将分离结果编码为 WAV 格式提供下载

## ⚠️ 限制说明

- **文件大小限制**: 建议 < 100MB (受浏览器内存限制)
- **处理时间**: 取决于设备性能和音频时长，手机端可能较慢
- **内存占用**: 大文件处理需要较多内存，建议桌面端使用
- **模型下载**: 首次使用需下载 45-120MB 模型文件
- **浏览器兼容**: 需要支持 WebAssembly、Web Audio API、OffscreenCanvas 的现代浏览器

## 🌐 浏览器兼容性

| 浏览器 | 支持版本 |
|--------|----------|
| Chrome | 89+ |
| Firefox | 78+ |
| Safari | 15+ |
| Edge | 89+ |

## 📝 许可证

MIT License - 可自由使用、修改、分发

## 🙏 致谢

- [Demucs](https://github.com/facebookresearch/demucs) - Facebook Research
- [Open-Unmix](https://github.com/sigsep/open-unmix) - SIGSEP
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) - Microsoft
- 模型托管于 [Hugging Face](https://huggingface.co/onnx-community)

---

**提示**: 如果遇到模型下载失败，可能是网络问题，请尝试刷新页面重试，或使用科学上网工具。