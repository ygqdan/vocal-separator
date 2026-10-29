/**
 * 音乐人声分离器 - 主应用逻辑
 * 基于 ONNX.js 在浏览器端运行 AI 模型
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';

// 配置 ONNX Runtime
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

// 模型配置
const MODEL_CONFIGS = {
    'htdemucs': {
        name: 'HTDemucs',
        sources: ['vocals', 'drums', 'bass', 'other'],
        url: 'https://huggingface.co/onnx-community/htdemucs_onnx/resolve/main/model.onnx',
        inputShape: [1, 2, 44100 * 10], // 10秒片段
        sampleRate: 44100,
        segmentLength: 44100 * 10,
        overlap: 0.25
    },
    'mdx-net': {
        name: 'MDX-Net',
        sources: ['vocals', 'accompaniment'],
        url: 'https://huggingface.co/onnx-community/mdx-net_onnx/resolve/main/model.onnx',
        inputShape: [1, 2, 44100 * 10],
        sampleRate: 44100,
        segmentLength: 44100 * 10,
        overlap: 0.1
    },
    'umxl': {
        name: 'UMX-L (Open-Unmix)',
        sources: ['vocals', 'drums', 'bass', 'other'],
        url: 'https://huggingface.co/onnx-community/open-unmix_onnx/resolve/main/umxl.onnx',
        inputShape: [1, 2, 44100 * 10],
        sampleRate: 44100,
        segmentLength: 44100 * 10,
        overlap: 0.1
    }
};

// 状态管理
const state = {
    audioFile: null,
    audioBuffer: null,
    selectedModel: 'htdemucs',
    session: null,
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
    elements.processBtn = document.getElementById('processBtn');
    elements.processProgress = document.getElementById('processProgress');
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

    // Label 原生处理点击触发文件选择，只需监听 input 的 change 事件
    elements.audioInput.addEventListener('change', handleFileSelect);

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
    const files = dt.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('audio/')) {
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

// 加载音频文件为 AudioBuffer
function loadAudioFile(file) {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.src = url;
    audio.addEventListener('canplaythrough', async () => {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioContext = new AudioContext({ sampleRate: 44100 });
            state.audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            URL.revokeObjectURL(url);
            showToast('音频加载完成', 'success');
        } catch (error) {
            console.error('音频解码失败:', error);
            showToast('音频解码失败', 'error');
            URL.revokeObjectURL(url);
        }
    });
    audio.addEventListener('error', () => {
        showToast('音频加载失败', 'error');
        URL.revokeObjectURL(url);
    });
}

// 模型选择处理
function setupModelSelection() {
    elements.modelRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.checked) {
                state.selectedModel = e.target.value;
                // 重置模型加载状态
                state.modelLoaded = false;
                state.session = null;
                loadModel();
            }
        });
    });
}

// 下载并加载模型
async function loadModel() {
    const config = MODEL_CONFIGS[state.selectedModel];
    elements.modelProgress.hidden = false;
    elements.modelProgressFill.style.width = '0%';
    elements.modelProgressText.textContent = `正在下载 ${config.name} 模型...`;

    try {
        // 创建推理会话
        // 使用进度回调模拟下载进度
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        // 模拟下载进度（实际 ONNX.js 不直接暴露下载进度）
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 15, 90);
            elements.modelProgressFill.style.width = `${progress}%`;
            elements.modelProgressText.textContent = `正在下载 ${config.name} 模型... ${Math.round(progress)}%`;
        }, 500);

        state.session = await ort.InferenceSession.create(config.url, sessionOptions);

        clearInterval(progressInterval);
        elements.modelProgressFill.style.width = '100%';
        elements.modelProgressText.textContent = '模型加载完成！';

        state.modelLoaded = true;
        setTimeout(() => {
            elements.modelProgress.hidden = true;
        }, 1000);

        showToast(`${config.name} 模型加载成功`, 'success');
    } catch (error) {
        console.error('模型加载失败:', error);
        elements.modelProgressText.textContent = '模型加载失败，请重试';
        elements.modelProgressFill.style.background = 'var(--color-error)';
        showToast('模型加载失败，请检查网络连接', 'error');
    }
}

// 音频处理主流程
async function processAudio() {
    if (!state.audioBuffer) {
        showToast('请先选择音频文件', 'error');
        return;
    }

    if (!state.modelLoaded || !state.session) {
        showToast('模型未加载完成', 'error');
        return;
    }

    state.isProcessing = true;
    elements.processBtn.disabled = true;
    elements.processBtn.querySelector('.btn-text').hidden = true;
    elements.processBtn.querySelector('.btn-loading').hidden = false;
    elements.processProgress.hidden = false;
    elements.processProgressFill.style.width = '0%';
    hideResults();

    try {
        const config = MODEL_CONFIGS[state.selectedModel];
        const results = await separateAudio(state.audioBuffer, config);
        state.results = results;
        displayResults(results, config);
        showToast('分离完成！', 'success');
    } catch (error) {
        console.error('处理失败:', error);
        showToast('处理失败: ' + error.message, 'error');
    } finally {
        state.isProcessing = false;
        elements.processBtn.disabled = false;
        elements.processBtn.querySelector('.btn-text').hidden = false;
        elements.processBtn.querySelector('.btn-loading').hidden = true;
        elements.processProgress.hidden = true;
    }
}

// 音频分离核心算法
async function separateAudio(audioBuffer, config) {
    const { sampleRate, segmentLength, overlap, sources } = config;
    const numChannels = audioBuffer.numberOfChannels;
    const totalLength = audioBuffer.length;

    // 重采样到模型要求的采样率 (如果需要)
    let inputBuffer = audioBuffer;
    if (audioBuffer.sampleRate !== sampleRate) {
        inputBuffer = await resampleAudio(audioBuffer, sampleRate);
    }

    // 准备输入数据
    const inputData = new Float32Array(numChannels * inputBuffer.length);
    for (let ch = 0; ch < numChannels; ch++) {
        inputData.set(inputBuffer.getChannelData(ch), ch * inputBuffer.length);
    }

    // 分段处理
    const hopSize = Math.floor(segmentLength * (1 - overlap));
    const numSegments = Math.ceil((inputBuffer.length - segmentLength) / hopSize) + 1;

    // 输出缓冲区
    const outputBuffers = {};
    sources.forEach(source => {
        outputBuffers[source] = new Float32Array(numChannels * inputBuffer.length);
    });

    // 窗函数 (Hann 窗)
    const window = new Float32Array(segmentLength);
    for (let i = 0; i < segmentLength; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (segmentLength - 1)));
    }

    const startTime = Date.now();

    for (let seg = 0; seg < numSegments; seg++) {
        const start = seg * hopSize;
        const end = Math.min(start + segmentLength, inputBuffer.length);
        const actualLength = end - start;

        // 准备分段输入
        const segmentInput = new Float32Array(numChannels * segmentLength);
        for (let ch = 0; ch < numChannels; ch++) {
            const channelData = inputBuffer.getChannelData(ch);
            segmentInput.set(channelData.slice(start, end), ch * segmentLength);
            // 零填充
            if (actualLength < segmentLength) {
                segmentInput.fill(0, ch * segmentLength + actualLength, (ch + 1) * segmentLength);
            }
        }

        // 应用窗函数
        for (let ch = 0; ch < numChannels; ch++) {
            for (let i = 0; i < segmentLength; i++) {
                segmentInput[ch * segmentLength + i] *= window[i];
            }
        }

        // 创建张量
        const tensor = new ort.Tensor('float32', segmentInput, [1, numChannels, segmentLength]);

        // 推理
        const feeds = { input: tensor };
        const results = await state.session.run(feeds);

        // 处理输出
        sources.forEach((source, idx) => {
            const outputTensor = results[`output_${idx}`] || results[Object.keys(results)[idx]];
            const outputData = outputTensor.data;

            // 叠加到输出缓冲区 (重叠相加)
            for (let ch = 0; ch < numChannels; ch++) {
                const outChannel = outputBuffers[source];
                const inChannel = outputData.subarray(ch * segmentLength, (ch + 1) * segmentLength);
                for (let i = 0; i < actualLength; i++) {
                    outChannel[ch * inputBuffer.length + start + i] += inChannel[i] * window[i];
                }
            }
        });

        // 更新进度
        const progress = ((seg + 1) / numSegments) * 100;
        elements.processProgressFill.style.width = `${progress}%`;
        elements.processProgressText.textContent = `正在分离... ${Math.round(progress)}% (${seg + 1}/${numSegments})`;

        const elapsed = (Date.now() - startTime) / 1000;
        const estimated = elapsed / (seg + 1) * numSegments;
        elements.processedTime.textContent = formatTime(elapsed);
        elements.totalTime.textContent = formatTime(estimated);

        // 让出主线程
        await new Promise(resolve => setTimeout(resolve, 0));
    }

    // 归一化重叠区域
    const overlapAddBuffer = new Float32Array(inputBuffer.length);
    for (let seg = 0; seg < numSegments; seg++) {
        const start = seg * hopSize;
        const end = Math.min(start + segmentLength, inputBuffer.length);
        for (let i = start; i < end; i++) {
            overlapAddBuffer[i] += window[i - start] * window[i - start];
        }
    }

    // 应用归一化并创建 AudioBuffer
    const resultBuffers = {};
    sources.forEach(source => {
        const outputData = outputBuffers[source];
        const normalizedData = new Float32Array(numChannels * inputBuffer.length);

        for (let ch = 0; ch < numChannels; ch++) {
            for (let i = 0; i < inputBuffer.length; i++) {
                const weight = overlapAddBuffer[i] > 0 ? overlapAddBuffer[i] : 1;
                normalizedData[ch * inputBuffer.length + i] = outputData[ch * inputBuffer.length + i] / weight;
            }
        }

        // 创建 AudioBuffer
        const ctx = new AudioContext({ sampleRate });
        const buffer = ctx.createBuffer(numChannels, inputBuffer.length, sampleRate);
        for (let ch = 0; ch < numChannels; ch++) {
            buffer.copyToChannel(normalizedData.subarray(ch * inputBuffer.length, (ch + 1) * inputBuffer.length), ch);
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

// 显示结果
function displayResults(results, config) {
    elements.resultsSection.hidden = false;

    // 显示/隐藏对应的结果卡片
    config.sources.forEach(source => {
        const card = document.getElementById(`${source}Card`);
        const audio = document.getElementById(`${source}Audio`);
        const download = document.getElementById(`${source}Download`);
        const duration = document.getElementById(`${source}Duration`);

        if (card && results[source]) {
            card.hidden = false;
            const blob = audioBufferToWav(results[source]);
            const url = URL.createObjectURL(blob);
            audio.src = url;
            audio.onloadedmetadata = () => {
                duration.textContent = formatTime(audio.duration);
            };
            download.href = url;
            download.download = `${source}_${state.audioFile.name.replace(/\.[^/.]+$/, '')}.wav`;
        } else if (card) {
            card.hidden = true;
        }
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

    // 自动加载默认模型
    await loadModel();
}

// 启动
document.addEventListener('DOMContentLoaded', init);