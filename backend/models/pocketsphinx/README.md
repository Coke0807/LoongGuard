# PocketSphinx 中文模型目录

PocketSphinx 指令识别所需模型。**本目录的模型与词典不入库**
（.gitignore 已排除 `zh-cn/` 与 `*.dict`），仓库内只保留语法文件与说明。

## 当前状态（2026-08-29，Windows 开发机已完成放置）

```
backend/models/pocketsphinx/
├── zh-cn/               # CMU Mandarin 声学模型（cmusphinx-zh-cn-5.2）
│   ├── feat.params
│   ├── feature_transform
│   ├── mdef
│   ├── means
│   ├── mixture_weights
│   ├── noisedict
│   ├── transition_matrices
│   └── variances
├── cmudict-cn.dict      # 中文词典（zh_cn.dic，139,872 条 + 追加"温湿度"）
├── commands.jsgf        # 指令语法（终结符均为词典存在的词条）
└── README.md
```

## 词典说明（重要）

`cmusphinx-zh-cn-5.2` 包的词典词条键为**中文字词**（如 `切换 w en1 …`），
不是拼音串。因此：

- `commands.jsgf`（中文版）是唯一可用的语法文件，其终结符已逐一核对
  为词典存在的词条；识别输出即中文词条序列。
- `VoiceConfig.sphinx_jsgf_path` 保持默认值即可，**无需**配置
  `LG_VOICE_SPHINX_JSGF_PATH`。
- 追加词条：指令需要的"温湿度"不在原词典中，已按"温+湿+度"音素拼接
  追加到 `cmudict-cn.dict` 末尾（`温湿度 uu un1 sh ix1 d u4`）。

## 全新环境重新放置步骤

1. 下载（SourceForge，约 54MB）：
   ```
   https://sourceforge.net/projects/cmusphinx/files/Acoustic and Language Models/Mandarin/cmusphinx-zh-cn-5.2.tar.gz
   ```
2. 解压后将 `zh_cn.cd_cont_5000/` 内容复制到 `zh-cn/`，
   `zh_cn.dic` 复制为 `cmudict-cn.dict`。
3. 在词典末尾追加：`温湿度 uu un1 sh ix1 d u4`

## 板端依赖安装（Loongnix）

```bash
sudo apt install pocketsphinx python3-pocketsphinx
```

（Windows 开发机直接 `pip install pocketsphinx` 即可，已验证 5.1.1 有 cp313 wheel）

## 验证

```bash
cd backend
python -c "
import sys; sys.path.insert(0, 'src')
from loongguard.voice import SphinxCommandRecognizer
r = SphinxCommandRecognizer(
    'models/pocketsphinx/zh-cn',
    'models/pocketsphinx/cmudict-cn.dict',
    'models/pocketsphinx/commands.jsgf')
print('available:', r.is_available)
"
```
