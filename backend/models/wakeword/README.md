# openWakeWord 唤醒词模型目录

自定义唤醒词（"小龙小龙"、"晓珑监护"）的 ONNX 模型（**不入库**，训练产出后放置）。

## 目录结构（放置完成后）

```
backend/models/wakeword/
└── my_wakeword.onnx    # 自训练唤醒词模型（~200KB）
```

## 训练（一次性工作，在 Windows 开发机或 Colab 上完成）

使用仓库自带训练脚本：

```bash
pip install openwakeword torch torchaudio
cd backend
python scripts/train_wakeword.py --help
```

训练流程（脚本已自动化）：
1. 用 Piper TTS 合成唤醒词正样本（"小龙小龙"、"晓珑监护"），
   或使用 `--clips` 指定真人录音目录（推荐，识别率更高）
2. 负样本使用 openWakeWord 自带增强数据集
3. 训练并导出 `my_wakeword.onnx`
4. 验证：安静环境唤醒率 > 95%、误唤醒 < 1 次/小时

详见脚本头部说明：`backend/scripts/train_wakeword.py`

## 板端部署

```bash
# Loongnix venv（龙芯 PyPI 源）
pip install openwakeword   # 自动拉取 onnxruntime loongarch64 wheel
```

模型放置后重启后端即可。缺模型时语音模块自动停用（日志有警告），
不影响检测/告警/家长端等其他功能。

## 注意

- openWakeWord 的 Speex 噪声抑制在 loong64 上不可用（可选增强，不影响推理）
- 修改唤醒词需重新训练模型并更新 `commands.jsgf` 无关（唤醒词不在指令语法内）
