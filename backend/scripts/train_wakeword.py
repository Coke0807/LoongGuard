#!/usr/bin/env python3
"""
openWakeWord 自定义唤醒词训练编排脚本（"小龙小龙" / "晓珑监护"）

设计动机：
    openWakeWord 官方训练流程为 notebook + train.py 三步流水线
    （见 https://github.com/dscripka/openWakeWord 的
    notebooks/automatic_model_training.ipynb）。本脚本把该流程
    固化为一键命令，产出 backend/models/wakeword/my_wakeword.onnx。

训练环境要求（重要）：
    - 仅支持 Linux（piper-sample-generator 的 TTS 依赖限制），
      推荐 Ubuntu/WSL2 或 Google Colab；不建议在 2K3000 板上训练
    - 依赖安装（官方 notebook 同款）：
        git clone https://github.com/rhasspy/piper-sample-generator
        wget -O piper-sample-generator/models/zh_CN-huayan-medium.pt \
            https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/zh_CN-huayan-medium.pt
        pip install piper-phonemize webrtcvad
        git clone https://github.com/dscripka/openwakeword
        pip install -e ./openwakeword
        pip install mutagen torchinfo torchmetrics speechbrain \
                    audiomentations torch-audiomentations acoustics \
                    datasets deep-phonemizer pronouncing

使用方式：
    # 全流程（生成正样本→增强→训练→安装到 backend/models/wakeword/）
    python scripts/train_wakeword.py --workspace ./wakeword_ws

    # 只生成/增强/训练单步
    python scripts/train_wakeword.py --workspace ./wakeword_ws --step generate
    python scripts/train_wakeword.py --workspace ./wakeword_ws --step augment
    python scripts/train_wakeword.py --workspace ./wakeword_ws --step train

数据资产（首次运行自动下载，可 --skip-download 跳过）：
    - MIT 房间脉冲响应（HuggingFace datasets）
    - 负样本特征 openwakeword_features_ACAV100M_2000_hrs_16bit.npy
    - 验证特征 validation_set_features.npy
    （噪声/音乐背景集体积较大，脚本给出下载指引但不强制）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("./wakeword_ws")

# 唤醒词（正样本短语）
TARGET_PHRASES = ["小龙小龙", "晓珑监护"]

# 预训练特征文件（HuggingFace 托管，官方训练流水线使用的负样本）
FEATURE_FILES = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    ),
    "validation_set_features.npy": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/validation_set_features.npy"
    ),
}

# openwakeword 训练脚本默认输出目录（相对训练时 CWD）
MODEL_OUTPUT_DIR = "my_custom_model"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] 已存在: {dest}")
        return
    print(f"[download] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 受控白名单源


def download_assets(workspace: Path) -> None:
    """下载负样本/验证特征（大文件，已存在则跳过）"""
    for name, url in FEATURE_FILES.items():
        download(url, workspace / name)


def write_training_config(workspace: Path, openwakeword_dir: Path) -> Path:
    """
    基于官方示例配置 custom_model.yml 生成中文唤醒词训练配置。

    优先直接修改官方模板（字段最全、注释最准）；
    模板缺失时用内置最小配置兜底。
    """
    template = openwakeword_dir / "openwakeword" / "examples" / "custom_model.yml"
    config_path = workspace / "loongguard_wakeword.yaml"

    import yaml  # 训练环境必然安装（openwakeword 训练依赖）

    if template.exists():
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
    else:
        config = {}

    config["target_phrase"] = TARGET_PHRASES
    config["model_name"] = "my_wakeword"
    config["n_samples"] = 4000          # 每个短语正样本量（官方建议数千级）
    config["n_samples_val"] = 1000      # 早停验证样本量
    config["steps"] = 30000             # 训练步数（数据量大时收益更高）
    config["target_accuracy"] = 0.7     # 官方默认推荐目标
    config["target_recall"] = 0.5
    # 背景噪声目录（如已下载 audioset/fma 则在此追加）
    background_paths = []
    for bg in ("audioset_16k", "fma"):
        if (workspace / bg).exists():
            background_paths.append(str(workspace / bg))
    config["background_paths"] = background_paths or ["./mit_rirs"]
    config["false_positive_validation_data_path"] = str(
        workspace / "validation_set_features.npy")
    config["feature_data_files"] = {
        "ACAV100M_sample": str(
            workspace / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    }

    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    print(f"[config] 训练配置已写入: {config_path}")
    return config_path


def run_train_step(openwakeword_dir: Path, config_path: Path, step: str) -> None:
    """调用官方 train.py 执行单步（generate/augment/train 全集）"""
    flag = {
        "generate": "--generate_clips",
        "augment": "--augment_clips",
        "train": "--train_model",
        "all": None,
    }[step]
    train_py = openwakeword_dir / "openwakeword" / "train.py"
    if not train_py.exists():
        sys.exit(f"[error] 找不到 {train_py}，请先: "
                 f"git clone https://github.com/dscripka/openwakeword "
                 f"并 pip install -e ./openwakeword")
    cmd = [sys.executable, str(train_py), "--training_config", str(config_path)]
    if flag:
        cmd.append(flag)
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=openwakeword_dir)


def install_model(workspace: Path) -> None:
    """把训练产出的 onnx 安装到 backend/models/wakeword/"""
    candidates = list(workspace.rglob("my_wakeword.onnx")) + \
        list(Path(MODEL_OUTPUT_DIR).glob("my_wakeword.onnx"))
    if not candidates:
        sys.exit("[error] 未找到训练产出 my_wakeword.onnx")
    src = candidates[0]
    dest = BACKEND_DIR / "models" / "wakeword" / "my_wakeword.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[install] 唤醒词模型已安装: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoongGuard 自定义唤醒词训练编排（需 Linux 训练环境）")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE),
                        help="训练工作区目录（数据/配置/产出存放处）")
    parser.add_argument("--openwakeword", default="./openwakeword",
                        help="openwakeword 仓库克隆路径")
    parser.add_argument("--step", default="all",
                        choices=["generate", "augment", "train", "all", "install"],
                        help="执行单步或全流程（默认 all）")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据资产下载（已手工准备时使用）")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    openwakeword_dir = Path(args.openwakeword).resolve()

    if sys.platform == "win32":
        print("[warn] Windows 原生环境不受官方支持（Piper TTS 限制），"
              "建议在 WSL2/Colab 中运行本脚本")

    if args.step == "install":
        install_model(workspace)
        return

    if not args.skip_download:
        download_assets(workspace)

    config_path = write_training_config(workspace, openwakeword_dir)

    if args.step == "all":
        run_train_step(openwakeword_dir, config_path, "generate")
        run_train_step(openwakeword_dir, config_path, "augment")
        run_train_step(openwakeword_dir, config_path, "train")
        install_model(workspace)
    else:
        run_train_step(openwakeword_dir, config_path, args.step)
        if args.step == "train":
            install_model(workspace)


if __name__ == "__main__":
    main()
