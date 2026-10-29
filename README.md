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

### 一键启动脚本

| 系统 | 脚本 | 说明 |
|------|------|------|
| Windows | `start.bat` | 双击运行；自动检测 Python / Node.js / PHP，自动打开浏览器 |
| macOS / Linux | `start.sh` | `chmod +x start.sh && ./start.sh` |

`start.bat` 支持指定端口：`start.bat 8080`；设置 `NO_OPEN=1` 可禁止自动打开浏览器。

## 📦 模型说明

| 模型 | 文件大小 | 分离轨道 | 说明 |
|------|----------|----------|------|
| **HTDemucs 完整版**（默认） | ~316MB | 鼓、贝斯、其他、人声（+ 自动合成伴奏） | fp32 权重，兼容性最好 |
| **HTDemucs 轻量版** | ~166MB | 同上 | fp16 存储权重，下载小一半，依赖 float16 算子支持 |

模型来自 [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx)（MIT），
输入/输出契约：

| Tensor | 名称 | 形状 | 说明 |
|---|---|---|---|
| 输入 | `mix` | `[1, 2, 343980]` | 44.1kHz 立体声，7.8 秒片段 |
| 输出 | `stems` | `[1, 4, 2, 343980]` | 依次是 `[drums, bass, other, vocals]` |

> ⚠️ **首次使用需下载模型文件**，建议在 WiFi 环境下使用。
> 加载顺序：**本地 `models/` 目录 → `hf-mirror.com` → `huggingface.co`**。
> 很多公司网络/浏览器代理会把 `hf-mirror.com` 跳转到的 `cas-bridge.xethub.hf.co` 拦掉
> （浏览器里只报一句 `Failed to fetch`，但命令行 `curl` 却能下），所以最稳的做法是用脚本提前下好：

```bat
:: Windows：下载到 models\ （默认完整版 316MB，加 fp16 参数下载轻量版 166MB）
download-model.bat
download-model.bat fp16
```

```bash
# macOS / Linux
mkdir -p models
curl -L -C - -o models/htdemucs.onnx \
  https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx
```

> 放了本地文件后，页面会自动优先使用它（不再走网络），刷新即可生效。

## 🛠 技术栈

- **ONNX Runtime Web** - 浏览器端 ONNX 模型推理（WASM 后端）
- **Web Audio API** - 音频解码、重采样、处理
- **原生 JavaScript (ES Modules)** - 无框架依赖，轻量快速
- **CSS 变量 + 现代 CSS** - 设计系统，支持深色模式

## 📁 项目结构

```
vocal-separator/
├── index.html           # 主页面
├── styles.css           # 样式文件
├── app.js               # 主逻辑（改动后请同步改 index.html 里的 app.js?v=）
├── start.bat            # Windows 本地启动脚本
├── start.sh             # macOS / Linux 本地启动脚本
├── download-model.bat   # Windows 模型下载脚本（下到 models\）
├── models/              # 本地模型目录（已在 .gitignore 里，不提交）
└── README.md            # 说明文档
```

## 🔧 核心原理

1. **音频加载** - 使用 Web Audio API 解码音频文件为 AudioBuffer
2. **重采样** - 统一重采样到模型要求的 44.1kHz
3. **分段推理** - 把音频切成 7.8 秒的重叠片段，逐段送入 ONNX 模型（`mix` → `stems`）
4. **重叠相加 (OLA)** - 使用 Hann 窗函数平滑拼接分段输出
5. **伴奏合成** - 用「原曲 − 人声」补齐伴奏轨
6. **导出下载** - 将分离结果编码为 WAV 格式提供下载

## ⚠️ 限制说明

- **处理时间**: 受浏览器 WASM 单线程限制，处理速度远慢于桌面版 Demucs。
  实测一首 3 分钟的歌可能需要十几分钟甚至更久，**建议先用 20–30 秒的片段验证效果**。
- **多线程**: 只有站点响应头带 `Cross-Origin-Opener-Policy: same-origin` 和
  `Cross-Origin-Embedder-Policy: require-corp`（即启用 SharedArrayBuffer）时，ONNX Runtime 才能多线程，
  速度可提升数倍。GitHub Pages 不支持自定义响应头。
- **内存占用**: 模型 + wasm 堆大约需要 1GB 以上内存，长音频还需要更多；建议桌面端使用。
- **音频时长**: 超过 10 分钟会直接拒绝处理。
- **浏览器兼容**: 需要支持 WebAssembly、Web Audio API 的现代浏览器。

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
- 模型托管于 [Hugging Face · StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx)

---

**提示**: 模型下载走 `hf-mirror.com`，国内一般可直连；若仍失败，见下方“常见问题”。

---

## 🐞 常见问题

### 点击上传框后“没反应”

已经修复。原因有两层：

1. **选完文件后页面毫无变化（原 bug 主因）**
   `app.js` 之前用 `import * as ort from '.../dist/ort.min.js'` 静态引入 ONNX Runtime。
   该文件是 UMD 构建、不是 ES Module，`import *` 拿到的命名空间是空的，`ort.env` 为 `undefined`，
   脚本在模块求值阶段就抛 `TypeError`，后面所有 `addEventListener` 都不会执行 ——
   文件选择框能打开，但选完文件后页面没有任何反应。
   现在改为动态加载 `dist/esm/ort.min.js`（真正的 ESM 构建），并带多 CDN 降级。

2. **CDN 不可达**
   原来的顶层静态 `import` 一旦失败，整个模块直接不执行，表现同样是“毫无反应”。
   现在依次尝试 jsDelivr / fastly.jsDelivr / unpkg / npmmirror，全部失败时会在页面顶部显示红色提示，
   并在点击“开始分离”时给出明确错误，而不是静默无响应。

顺带把“打开页面就自动下载 100MB+ 模型”改成了点击“开始分离”时才懒加载，避免首屏看起来像卡死。

### 页面顶部出现红色提示

说明 `app.js` 没有正常执行：通常是广告拦截插件拦掉了脚本，或静态资源加载失败。刷新页面即可重试。

### 点击“开始分离”提示“模型加载失败：Failed to fetch”

原代码里 `MODEL_CONFIGS` 指向的是**不存在**的仓库（`onnx-community/htdemucs_onnx`），而且输入输出契约也不对
（Demucs 的 ONNX 导出需要 `mix`/`stems` 而不是随便一个 `input`），所以一定失败。

现在已换成真实可用的模型 [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx)：

- 默认从 `hf-mirror.com` 拉取（国内可直连），失败回退 `huggingface.co`；
- 下载时有**真实进度**（MB 数 + 百分比），不再是假动画；
- 网络完全不通时，可以把模型文件下载到本地，用「选择本地模型文件 (.onnx)」直接加载：

```bash
# 任意能联网的机器上执行（完整版 316MB）
curl -L -o htdemucs.onnx \
  https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx

# 或者轻量版（166MB，fp16 权重）
curl -L -o htdemucs_fp16weights.onnx \
  https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx
```

如果仍然报 `Failed to fetch`，先分清是谁在拦：

```bash
# 命令行能下（200/206）但浏览器不能 —— 说明是浏览器代理/扩展在拦
curl.exe -I -L --max-time 20 \
  https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx
```

实测遇到过的情况：系统代理放行 `hf-mirror.com` 的 302，但把跳转目标
`cas-bridge.xethub.hf.co` 拦掉，浏览器 fetch 直接就 `Failed to fetch`（818ms 失败），
而 `curl` 走同一个代理却正常。这种“浏览器被拦、命令行能下”的场景，用
`download-model.bat` 把文件放到 `models/` 是最省事的解法——页面会优先读本地文件，根本不发网络请求。

### 页面还显示旧的行为 / 改了代码不生效

本地用 `python -m http.server` 时浏览器可能缓存 `app.js`。`index.html` 里给脚本加了
`?v=4` 版本号，**改动 app.js 后请把这个数字加 1**；或者直接按 `Ctrl+F5` 强制刷新。

### 加载模型时浏览器报 `Aborted()`

这基本是 **WASM 内存不够**：htdemucs 的推理会话本身要占用约 1GB 内存，加上 300MB 模型文件，
峰值需求在 1.5GB 以上。内存吃紧时会直接 abort，而不是给出友好提示。

处理办法：关闭其他标签页和占内存的程序后重试；或者换台内存更宽裕的机器。
页面本身也会提示“浏览器内存不足”，但浏览器抛出的 `Aborted()` 无法拿到更多细节。

### 分离很慢 / 风扇狂转

浏览器里是 WASM **单线程**推理，htdemucs 在 CPU 上大约每 7.8 秒片段需要几十秒。
一首 3 分钟的歌需要切 ~24 段，所以整体可能要十几分钟，这是当前方案的物理限制，不是卡死。
建议先用 20–30 秒的片段验证效果；正式处理请用桌面版 Demucs 或带 GPU 的服务端。
