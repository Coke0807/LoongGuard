# AI 语音交互项目部署文档

> ⚠️ **已废弃（2026-08-29）**：本文记录的 vosk/FunASR/sherpa 虚拟机部署实验
> 未被现行方案采用。现行语音方案为 openWakeWord（唤醒）+ PocketSphinx（指令），
> 见 [docs/03-语音与音频/语音唤醒与指令实现.md](../03-语音与音频/语音唤醒与指令实现.md)。
> 下文仅作历史踩坑记录保留。

> 在龙芯虚拟机中部署语音识别（ASR）与语音播报（TTS）的完整记录与踩坑总结。

## 1. 虚拟机配置

启动脚本 `start.bat`，关键注意点：

- **内存 `-m 32G`**：32G 内存仅适合实体 64G 及以上主机。龙芯虚拟机编译语音库非常吃内存，内存不足会编译直接崩溃。
- **音频 `intel-hda + hda-duplex`**：完整双工声卡（麦克风输入 + 扬声器输出），刚好匹配项目语音采集 + 语音播报需求。

## 2. 语音播报部分：espeak

系统自带的 espeak，中文有点卡，英文很流畅。

安装指令：

```bash
sudo apt update
sudo apt install espeak espeak-data libespeak1 libespeak-dev portaudio19-dev
```

测试指令：

```bash
# 英文测试
espeak "Hello, this is voice test"

# 中文测试
espeak -v zh "你好，龙芯虚拟机语音测试成功"
```

## 3. 语音识别部分

### 3.1 vosk 模型

- 在 x8s 测试上识别准确率还可以。
- 但没有龙架构官方预编译 wheel 包，不能直接 `pip install vosk`，需要源码在虚拟环境编译。

**X8s 测试代码：**

（见原文档 image3、image4）

**龙架构上源码编译：**

（见原文档 image5、image6）

**遇到的困难：** Vosk 编译依赖 Kaldi 语音工具库，Deepin 龙芯源没有预装 kaldi 开发包，cmake 找不到 `kaldi-config.cmake` 直接终止编译。

### 3.2 sherpa 模型

（见原文档 image8、image9）

**出现的问题：** cmake 编译过程老是卡死。

### 3.3 SenceVoice 模型

（见原文档 image11、image12、image13）

**问题：** 不存在 packaging 离线 whl 安装包。

## 4. 总结

目前仍在尝试编译 vosk 源码适配龙架构，`long4` 为 **vosk + espeak** 虚拟机端在环境搭好的情况下的测试代码。

目前主要的问题：很多模型大多数都没有适配龙架构的包和依赖，需要进行源码编译。

> 注：原文档包含大量截图（image1~image14），此处仅归纳文字要点。如需查看截图，请查阅 `docs/archive/AI语音交互项目部署文档.docx` 原文件。