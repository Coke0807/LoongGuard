#!/usr/bin/env python3
"""
LoongGuard 远程执行脚本
在 Windows 本地编辑，通过 SSH 在 LoongArch 设备上执行命令

使用方法：
    # 同步代码并运行测试
    python scripts/remote_exec.py test

    # 同步代码并运行特定测试
    python scripts/remote_exec.py test --file tests/unit/test_detection.py

    # 同步代码并启动服务
    python scripts/remote_exec.py run

    # 仅同步代码
    python scripts/remote_exec.py sync

    # 在远程主机执行自定义命令
    python scripts/remote_exec.py exec --cmd "python -c 'import onnxruntime; print(onnxruntime.get_available_providers())'"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import paramiko

# ============================================================================
# 配置
# ============================================================================

REMOTE_CONFIG = {
    "host": os.environ.get("LG_SSH_HOST", ""),
    "port": int(os.environ.get("LG_SSH_PORT", "22")),
    "username": os.environ.get("LG_SSH_USER", ""),
    "password": os.environ.get("LG_SSH_PASS", ""),
    "remote_dir": "/home/cc/project/LoongGuard/",
}

# ============================================================================
# SSH 执行器
# ============================================================================

class RemoteExecutor:
    """远程命令执行器"""

    def __init__(self, config: dict):
        self.config = config
        self.ssh: Optional[paramiko.SSHClient] = None
        self.logger = logging.getLogger("RemoteExec")

    def connect(self) -> bool:
        """建立 SSH 连接"""
        # 验证必填连接参数
        host = self.config["host"]
        username = self.config["username"]
        if not host or not username:
            raise RuntimeError(
                "SSH 连接参数未配置。请设置环境变量 LG_SSH_HOST, LG_SSH_USER, LG_SSH_PASS，"
                "或复制 .env.example 为 .env 并填写。"
            )
        try:
            self.logger.info(f"连接 {self.config['host']}:{self.config['port']}...")
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.config["host"],
                port=self.config["port"],
                username=self.config["username"],
                password=self.config["password"],
                timeout=10,
            )
            self.logger.info("SSH 连接成功")
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            return False

    def disconnect(self):
        """关闭连接"""
        if self.ssh:
            self.ssh.close()

    def execute(self, command: str, check: bool = True) -> tuple[int, str, str]:
        """
        执行远程命令

        Args:
            command: 要执行的命令
            check: 是否检查退出码

        Returns:
            (returncode, stdout, stderr)
        """
        if not self.ssh:
            raise RuntimeError("SSH 未连接")

        self.logger.debug(f"执行: {command}")

        # 切换到项目目录再执行命令
        full_command = f"cd {self.config['remote_dir']} && {command}"

        stdin, stdout, stderr = self.ssh.exec_command(full_command)

        exit_code = stdout.channel.recv_exit_status()
        stdout_str = stdout.read().decode("utf-8", errors="replace")
        stderr_str = stderr.read().decode("utf-8", errors="replace")

        if check and exit_code != 0:
            self.logger.error(f"命令失败 (exit={exit_code}): {command}")
            if stderr_str:
                self.logger.error(f"stderr:\n{stderr_str}")

        return exit_code, stdout_str, stderr_str

# ============================================================================
# 命令处理器
# ============================================================================

def cmd_sync(executor: RemoteExecutor, args: argparse.Namespace) -> int:
    """同步代码到远程主机"""
    print("=" * 60)
    print("同步代码到 LoongArch 设备...")
    print("=" * 60)

    # 调用 sync.py
    sync_script = Path(__file__).parent / "sync.py"
    ret = os.system(f"{sys.executable} {sync_script}")

    if ret != 0:
        print("同步失败!")
        return 1

    print("同步完成!")
    return 0

def cmd_test(executor: RemoteExecutor, args: argparse.Namespace) -> int:
    """在远程主机运行测试"""
    print("=" * 60)
    print("在 LoongArch 设备运行测试...")
    print("=" * 60)

    # 先同步代码
    ret = cmd_sync(executor, args)
    if ret != 0:
        return ret

    # 构建测试命令
    test_cmd = "python -m pytest tests/ -v"
    if args.file:
        test_cmd = f"python -m pytest {args.file} -v"
    if args.keyword:
        test_cmd += f" -k {args.keyword}"

    # 执行测试
    print(f"\n执行: {test_cmd}")
    print("-" * 60)

    exit_code, stdout, stderr = executor.execute(test_cmd)

    # 输出结果
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    print("-" * 60)
    if exit_code == 0:
        print("测试通过!")
    else:
        print(f"测试失败 (exit={exit_code})")

    return exit_code

def cmd_run(executor: RemoteExecutor, args: argparse.Namespace) -> int:
    """在远程主机启动服务"""
    print("=" * 60)
    print("在 LoongArch 设备启动 LoongGuard 服务...")
    print("=" * 60)

    # 先同步代码
    ret = cmd_sync(executor, args)
    if ret != 0:
        return ret

    # 启动服务（统一入口 run.py）
    run_cmd = "python run.py"
    if args.config:
        run_cmd += f" {args.config}"

    print(f"\n执行: {run_cmd}")
    print("-" * 60)
    print("按 Ctrl+C 停止服务")
    print("-" * 60)

    try:
        exit_code, stdout, stderr = executor.execute(run_cmd, check=False)
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        return exit_code
    except KeyboardInterrupt:
        print("\n服务已停止")
        return 0

def cmd_exec(executor: RemoteExecutor, args: argparse.Namespace) -> int:
    """执行自定义命令"""
    if not args.cmd:
        print("错误: 必须指定 --cmd 参数")
        return 1

    print("=" * 60)
    print(f"在 LoongArch 设备执行: {args.cmd}")
    print("=" * 60)

    exit_code, stdout, stderr = executor.execute(args.cmd, check=False)

    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    return exit_code

def cmd_check_env(executor: RemoteExecutor, args: argparse.Namespace) -> int:
    """检查远程环境配置"""
    print("=" * 60)
    print("检查 LoongArch 环境配置...")
    print("=" * 60)

    checks = [
        ("Python 版本", "python3 --version"),
        ("pip 版本", "pip3 --version"),
        ("ONNX Runtime", "python3 -c 'import onnxruntime; print(onnxruntime.__version__)'"),
        ("OpenCV", "python3 -c 'import cv2; print(cv2.__version__)'"),
        ("NumPy", "python3 -c 'import numpy; print(numpy.__version__)'"),
        ("ONNX 执行提供程序", "python3 -c 'import onnxruntime; print(onnxruntime.get_available_providers())'"),
        ("V4L2 设备", "ls -la /dev/video* 2>/dev/null || echo '无 V4L2 设备'"),
        ("GPU 信息", "lspci | grep -i vga || echo '无 GPU 信息'"),
        ("内核版本", "uname -a"),
        ("架构", "uname -m"),
    ]

    for name, cmd in checks:
        print(f"\n[ {name} ]")
        exit_code, stdout, stderr = executor.execute(cmd, check=False)
        if stdout:
            print(stdout.strip())
        if stderr and exit_code != 0:
            print(f"  (警告: {stderr.strip()})")

    return 0

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoongGuard 远程执行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 同步并运行所有测试
    python scripts/remote_exec.py test

    # 运行特定测试文件
    python scripts/remote_exec.py test --file tests/unit/test_detection.py

    # 运行匹配关键字的测试
    python scripts/remote_exec.py test --keyword "test_infer"

    # 启动服务
    python scripts/remote_exec.py run

    # 检查远程环境
    python scripts/remote_exec.py check-env

    # 执行自定义命令
    python scripts/remote_exec.py exec --cmd "nvidia-smi"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # sync 命令
    subparsers.add_parser("sync", help="同步代码到远程主机")

    # test 命令
    test_parser = subparsers.add_parser("test", help="在远程主机运行测试")
    test_parser.add_argument("--file", type=str, help="指定测试文件")
    test_parser.add_argument("--keyword", "-k", type=str, help="测试关键字过滤")

    # run 命令
    run_parser = subparsers.add_parser("run", help="在远程主机启动服务")
    run_parser.add_argument("--config", type=str, help="配置文件路径")

    # exec 命令
    exec_parser = subparsers.add_parser("exec", help="执行自定义命令")
    exec_parser.add_argument("--cmd", type=str, required=True, help="要执行的命令")

    # check-env 命令
    subparsers.add_parser("check-env", help="检查远程环境配置")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # 创建执行器
    executor = RemoteExecutor(REMOTE_CONFIG)

    # sync 和 check-env 不需要连接（sync 内部处理连接）
    if args.command in ("sync",):
        return cmd_sync(executor, args)

    # 其他命令需要 SSH 连接
    if not executor.connect():
        return 1

    try:
        if args.command == "test":
            return cmd_test(executor, args)
        elif args.command == "run":
            return cmd_run(executor, args)
        elif args.command == "exec":
            return cmd_exec(executor, args)
        elif args.command == "check-env":
            return cmd_check_env(executor, args)
        else:
            print(f"未知命令: {args.command}")
            return 1
    finally:
        executor.disconnect()

if __name__ == "__main__":
    sys.exit(main())
