#!/usr/bin/env python3
"""
LoongGuard 项目同步脚本
将本地开发文件同步到远程龙芯 3A5000 主机

使用方法：
    python scripts/sync.py
    python scripts/sync.py --dry-run  # 仅预览，不实际传输
"""

import os
import sys
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import paramiko

# ============================================================================
# 配置管理（集中定义，便于修改）
# ============================================================================

SYNC_CONFIG: Dict = {
    "remote": {
        "host": os.environ.get("LG_SSH_HOST", ""),
        "port": int(os.environ.get("LG_SSH_PORT", "22")),
        "username": os.environ.get("LG_SSH_USER", ""),
        "password": os.environ.get("LG_SSH_PASS", ""),
    },
    "remote_dir": "/home/cc/project/LoongGuard/",
    "sync_dirs": ["src", "config", "tests", "scripts", "models", "docs"],
    "sync_files": ["requirements.txt", "CLAUDE.md", "项目背景.md"],
    "always_ignore_dirs": {".git", "__pycache__", ".pytest_cache", "venv", ".venv", ".mypy_cache", ".ruff_cache", "logs"},
    "always_ignore_files": {"*.pyc", "*.pyo", "*.pyd", ".env.local", ".env"},
    # models/ 在 .gitignore 中被排除（避免提交大文件），但同步到龙芯时必须包含
    # tests/mock_ 被 tests/*.mp4 排除，但单元测试依赖 mock 视频
    "force_include_prefixes": ["models/", "tests/mock_"],
    "local_root": None,
}

# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """配置日志系统，同时输出到控制台和文件（延迟创建目录）"""
    logger = logging.getLogger("LoongGuard-Sync")
    logger.setLevel(logging.DEBUG)

    # 控制台处理器 - 始终可用
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 文件处理器 - 延迟初始化，首次写入时再创建目录
    try:
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"sync_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
    except OSError:
        logger.warning(f"无法创建日志目录 {log_dir}，日志仅输出到控制台")

    return logger

logger = setup_logging()

# ============================================================================
# .gitignore 解析器
# ============================================================================

class GitignoreParser:
    """
    解析 .gitignore 文件并提供路径匹配功能
    
    核心逻辑：
    - 读取项目根目录的 .gitignore 文件
    - 使用 pathspec 库进行模式匹配（如果可用）
    - 否则使用简化的 glob 匹配
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.patterns: List[str] = []
        self._load_gitignore()
    
    def _load_gitignore(self) -> None:
        """加载 .gitignore 文件中的规则"""
        gitignore_path = self.root_dir / ".gitignore"
        if not gitignore_path.exists():
            logger.debug("未找到 .gitignore 文件，跳过加载")
            return

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if line and not line.startswith("#"):
                        self.patterns.append(line)
            logger.debug(f"从 .gitignore 加载了 {len(self.patterns)} 条规则")
        except Exception as e:
            logger.warning(f"读取 .gitignore 失败: {e}")
    
    def _normalize_pattern(self, pattern: str) -> str:
        """标准化模式字符串"""
        # 移除前导斜杠
        if pattern.startswith("/"):
            pattern = pattern[1:]
        # 确保目录模式正确
        if pattern.endswith("/"):
            pattern = pattern[:-1]
        return pattern
    
    def is_ignored(self, rel_path: str, is_dir: bool = False) -> bool:
        """
        检查路径是否应该被忽略

        Args:
            rel_path: 相对于项目根目录的路径
            is_dir: 是否为目录

        Returns:
            True 表示应该忽略
        """
        if not self.patterns:
            return False

        rel_path = rel_path.replace("\\", "/")

        # 尝试使用 pathspec 库（更准确，支持 ! 反向规则）
        try:
            import pathspec
            spec = pathspec.PathSpec.from_lines("gitignore", self.patterns)
            return spec.match_file(rel_path)
        except ImportError:
            pass

        # 简化的 glob 匹配（回退方案，支持 ! 反向规则）
        from fnmatch import fnmatch

        path_parts = rel_path.split("/")
        ignored = False

        for pattern in self.patterns:
            is_negation = pattern.startswith("!")
            check_pattern = pattern[1:] if is_negation else pattern
            check_pattern = self._normalize_pattern(check_pattern)

            matched = False
            # 直接匹配文件名
            if fnmatch(rel_path, check_pattern):
                matched = True
            # 匹配路径中的任意部分
            elif fnmatch(path_parts[-1], check_pattern):
                matched = True
            # 匹配目录
            elif is_dir and fnmatch(rel_path + "/*", check_pattern + "/*"):
                matched = True

            if matched:
                if is_negation:
                    ignored = False  # ! 规则取消忽略
                else:
                    ignored = True

        return ignored

# ============================================================================
# 文件哈希计算（用于差异比较）
# ============================================================================

def calculate_file_hash(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值
    
    用途：比较本地和远程文件是否相同，避免不必要的传输
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希失败 {file_path}: {e}")
        return ""

# ============================================================================
# 同步统计信息
# ============================================================================

class SyncStats:
    """跟踪同步过程的统计信息"""
    
    def __init__(self):
        self.files_transferred = 0
        self.files_skipped = 0
        self.files_ignored = 0
        self.directories_created = 0
        self.errors: List[Tuple[str, str]] = []  # (文件路径, 错误信息)
        self.start_time = datetime.now()
    
    def add_error(self, file_path: str, error: str):
        self.errors.append((file_path, error))
    
    def print_summary(self):
        """打印同步结果摘要"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("同步完成!")
        print("=" * 60)
        print(f"传输文件数:     {self.files_transferred}")
        print(f"跳过文件数:     {self.files_skipped}")
        print(f"忽略文件数:     {self.files_ignored}")
        print(f"创建目录数:     {self.directories_created}")
        print(f"错误数量:       {len(self.errors)}")
        print(f"耗时:           {duration:.2f} 秒")
        
        if self.errors:
            print("\n错误详情:")
            for file_path, error in self.errors:
                print(f"  - {file_path}: {error}")
        
        print("=" * 60)
        
        # 记录到日志
        logger.info(f"同步完成 - 传输:{self.files_transferred}, 跳过:{self.files_skipped}, "
                    f"忽略:{self.files_ignored}, 错误:{len(self.errors)}, 耗时:{duration:.2f}s")

# ============================================================================
# 核心同步逻辑
# ============================================================================

class LoongGuardSyncer:
    """
    LoongGuard 项目同步器
    
    职责：
    - 建立 SSH/SFTP 连接
    - 扫描本地文件并应用过滤规则
    - 比较文件差异，仅传输修改过的文件
    - 记录传输日志和统计信息
    """
    
    def __init__(self, config: Dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.ssh: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        self.gitignore_parser: Optional[GitignoreParser] = None
        self.stats = SyncStats()
        self.remote_file_hashes: Dict[str, str] = {}
        
        # 确定本地项目根目录
        if config["local_root"]:
            self.local_root = Path(config["local_root"])
        else:
            # 默认为脚本所在目录的父目录
            self.local_root = Path(__file__).parent.parent
        
        logger.info(f"本地项目根目录: {self.local_root}")
        logger.info(f"远程目标路径: {config['remote_dir']}")
        logger.info(f"同步模式: {'预览' if dry_run else '实际传输'}")
    
    def connect(self) -> bool:
        """
        建立 SSH 连接

        认证优先级：SSH 密钥 -> 密码（支持环境变量覆盖）

        Returns:
            True 表示连接成功
        """
        remote = self.config["remote"]

        # 验证必填连接参数
        host = os.environ.get("LG_SSH_HOST", remote.get("host", ""))
        username = os.environ.get("LG_SSH_USER", remote.get("username", ""))
        if not host or not username:
            raise RuntimeError(
                "SSH 连接参数未配置。请设置环境变量 LG_SSH_HOST, LG_SSH_USER, LG_SSH_PASS，"
                "或复制 .env.example 为 .env 并填写。"
            )

        # 环境变量覆盖（遵循 12-Factor App，密码不硬编码）
        port = int(os.environ.get("LG_SSH_PORT", remote.get("port", 22)))
        password = os.environ.get("LG_SSH_PASS", remote.get("password", ""))
        key_file = os.environ.get("LG_SSH_KEY", "")  # SSH 私钥路径

        try:
            logger.info(f"正在连接 {host}:{port}...")

            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": 10,
            }

            # 优先使用 SSH 密钥
            if key_file and Path(key_file).exists():
                connect_kwargs["key_filename"] = key_file
                logger.info(f"使用 SSH 密钥认证: {key_file}")
            elif password:
                connect_kwargs["password"] = password
            else:
                logger.error("未提供认证信息：设置 LG_SSH_PASS 环境变量或 LG_SSH_KEY")
                return False

            self.ssh.connect(**connect_kwargs)
            
            self.sftp = self.ssh.open_sftp()
            logger.info("SSH 连接建立成功")
            
            # 初始化远程文件哈希缓存
            self._load_remote_hashes()
            
            return True
            
        except paramiko.AuthenticationException:
            logger.error("认证失败: 用户名或密码错误")
            self.stats.add_error("SSH连接", "认证失败")
            return False
        except paramiko.SSHException as e:
            logger.error(f"SSH 连接错误: {e}")
            self.stats.add_error("SSH连接", str(e))
            return False
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.stats.add_error("SSH连接", str(e))
            return False
    
    def disconnect(self):
        """关闭 SSH 连接"""
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        logger.debug("SSH 连接已关闭")
    
    def _load_remote_hashes(self):
        """
        加载远程文件的哈希缓存
        
        优化策略：
        - 在远程主机维护一个哈希文件（.sync_hashes.json）
        - 避免每次同步都重新计算所有远程文件的哈希
        """
        remote_hash_file = self.config["remote_dir"] + ".sync_hashes.json"
        
        try:
            # 尝试读取远程哈希缓存
            with self.sftp.open(remote_hash_file, "r") as f:
                import json
                self.remote_file_hashes = json.loads(f.read().decode("utf-8"))
            logger.debug(f"加载了 {len(self.remote_file_hashes)} 个远程文件哈希")
        except Exception:
            logger.debug("远程哈希缓存不存在或无法读取，将重新计算")
            self.remote_file_hashes = {}
    
    def _save_remote_hashes(self):
        """保存远程文件哈希缓存"""
        if not self.remote_file_hashes:
            return
        
        remote_hash_file = self.config["remote_dir"] + ".sync_hashes.json"
        
        try:
            import json
            data = json.dumps(self.remote_file_hashes, indent=2)
            with self.sftp.open(remote_hash_file, "w") as f:
                f.write(data.encode("utf-8"))
            logger.debug("远程哈希缓存已更新")
        except Exception as e:
            logger.warning(f"保存远程哈希缓存失败: {e}")
    
    def _ensure_remote_dir(self, remote_path: str):
        """
        确保远程目录存在，不存在则递归创建
        
        Args:
            remote_path: 远程目录的绝对路径
        """
        if self.dry_run:
            logger.debug(f"[预览] 将创建目录: {remote_path}")
            self.stats.directories_created += 1
            return
        
        try:
            # 检查目录是否已存在
            self.sftp.stat(remote_path)
        except FileNotFoundError:
            # 递归创建父目录
            parent = str(Path(remote_path).parent)
            if parent and parent != remote_path:
                self._ensure_remote_dir(parent)
            
            try:
                self.sftp.mkdir(remote_path)
                self.stats.directories_created += 1
                logger.debug(f"创建目录: {remote_path}")
            except IOError as e:
                # 目录可能已被其他进程创建
                if "exists" not in str(e).lower():
                    raise
    
    def _should_ignore(self, rel_path: str, is_dir: bool = False) -> bool:
        """
        检查路径是否应该被忽略

        过滤优先级：
        1. force_include 白名单（最高优先级，覆盖所有忽略规则）
        2. 始终忽略的目录/文件模式
        3. .gitignore 规则
        """
        rel_path_normalized = rel_path.replace("\\", "/")

        # 白名单优先：models/ 等目录在 .gitignore 中被排除，
        # 但同步到龙芯时必须包含，强制放行
        for prefix in self.config.get("force_include_prefixes", []):
            if rel_path_normalized.startswith(prefix):
                return False

        path_parts = Path(rel_path).parts
        
        # 检查始终忽略的目录
        if is_dir:
            dir_name = Path(rel_path).name
            if dir_name in self.config["always_ignore_dirs"]:
                return True
        else:
            # 检查文件是否在忽略的目录中
            for part in path_parts:
                if part in self.config["always_ignore_dirs"]:
                    return True
            
            # 检查始终忽略的文件模式
            from fnmatch import fnmatch
            file_name = Path(rel_path).name
            for pattern in self.config["always_ignore_files"]:
                if fnmatch(file_name, pattern):
                    return True
        
        # 检查 .gitignore 规则
        if self.gitignore_parser and self.gitignore_parser.is_ignored(rel_path, is_dir):
            return True
        
        return False
    
    def _is_file_changed(self, local_path: str, rel_path: str) -> bool:
        """
        检查文件是否已修改

        比较策略：
        1. 先比较文件大小（零开销）
        2. 小文件（<5MB）大小相同时比较 MD5 哈希
        3. 大文件（>=5MB）仅靠大小判断，避免通过 SFTP 下载整文件做哈希
        """
        # dry-run 模式无 SFTP 连接，保守地认为所有文件都需要传输
        if self.sftp is None:
            return True

        remote_path = self.config["remote_dir"] + rel_path.replace("\\", "/")

        try:
            # 获取本地文件信息
            local_stat = os.stat(local_path)
            local_size = local_stat.st_size

            # 获取远程文件信息
            try:
                remote_stat = self.sftp.stat(remote_path)
            except FileNotFoundError:
                return True  # 远程文件不存在，需要传输

            # 比较文件大小
            if local_size != remote_stat.st_size:
                return True

            # 大文件仅靠大小判断（避免通过 SFTP 下载 10MB+ 做哈希）
            LARGE_FILE_THRESHOLD = 5 * 1024 * 1024  # 5 MB
            if local_size > LARGE_FILE_THRESHOLD:
                return False

            # 小文件大小相同时，比较 MD5 哈希
            local_hash = calculate_file_hash(local_path)

            # 检查缓存
            if rel_path in self.remote_file_hashes:
                if local_hash == self.remote_file_hashes[rel_path]:
                    return False

            # 缓存未命中，计算远程文件哈希（仅小文件，可接受）
            remote_hash = self._calculate_remote_hash(remote_path)
            self.remote_file_hashes[rel_path] = remote_hash

            return local_hash != remote_hash

        except Exception as e:
            logger.debug(f"比较文件差异时出错 {rel_path}: {e}")
            return True  # 出错时保守地传输文件

    def _calculate_remote_hash(self, remote_path: str) -> str:
        """分块计算远程文件 MD5 哈希，避免一次性加载大文件到内存"""
        h = hashlib.md5()
        try:
            with self.sftp.open(remote_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.debug(f"计算远程文件哈希失败 {remote_path}: {e}")
            return ""
    
    def _transfer_file(self, local_path: str, rel_path: str):
        """
        传输单个文件到远程主机

        Args:
            local_path: 本地文件的绝对路径
            rel_path: 相对于项目根目录的路径
        """
        remote_path = self.config["remote_dir"] + rel_path.replace("\\", "/")
        remote_dir = str(Path(remote_path).parent)

        if self.dry_run:
            size_kb = os.path.getsize(local_path) / 1024
            logger.debug(f"[预览] 将传输: {rel_path} ({size_kb:.0f} KB)")
            self.stats.files_transferred += 1
            return

        try:
            # 确保远程目录存在
            self._ensure_remote_dir(remote_dir)

            # 大文件显示进度
            file_size = os.path.getsize(local_path)
            LARGE_THRESHOLD = 1024 * 1024  # 1 MB

            if file_size > LARGE_THRESHOLD:
                size_mb = file_size / 1024 / 1024
                logger.info(f"正在传输: {rel_path} ({size_mb:.1f} MB) ...")
                self.sftp.put(local_path, remote_path, callback=self._progress_callback)
                # 换行结束进度显示
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                self.sftp.put(local_path, remote_path)

            # 更新哈希缓存
            local_hash = calculate_file_hash(local_path)
            self.remote_file_hashes[rel_path] = local_hash

            self.stats.files_transferred += 1
            logger.info(f"已传输: {rel_path}")

        except Exception as e:
            logger.error(f"传输失败 {rel_path}: {e}")
            self.stats.add_error(rel_path, str(e))

    @staticmethod
    def _progress_callback(bytes_transferred: int, total_bytes: int) -> None:
        """SFTP 传输进度回调（仅大文件使用）"""
        if total_bytes > 0:
            pct = bytes_transferred * 100 // total_bytes
            mb_done = bytes_transferred / 1024 / 1024
            mb_total = total_bytes / 1024 / 1024
            sys.stdout.write(f"\r  进度: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)")
            sys.stdout.flush()
    
    def scan_and_sync(self):
        """
        扫描本地文件并同步到远程主机
        
        流程：
        1. 加载 .gitignore 规则
        2. 遍历配置的同步目录
        3. 应用过滤规则
        4. 比较文件差异
        5. 传输修改过的文件
        """
        # 初始化 .gitignore 解析器
        self.gitignore_parser = GitignoreParser(self.local_root)
        
        logger.info("开始扫描本地文件...")
        
        # 收集所有需要同步的文件
        files_to_sync: List[Tuple[str, str]] = []  # (本地路径, 相对路径)
        
        # 同步指定的目录
        for dir_name in self.config["sync_dirs"]:
            dir_path = self.local_root / dir_name
            if not dir_path.exists():
                logger.warning(f"目录不存在: {dir_name}")
                continue
            
            self._scan_directory(dir_path, dir_name, files_to_sync)
        
        # 同步指定的文件
        for file_name in self.config["sync_files"]:
            file_path = self.local_root / file_name
            if file_path.exists() and not self._should_ignore(file_name):
                files_to_sync.append((str(file_path), file_name))
        
        logger.info(f"扫描完成，发现 {len(files_to_sync)} 个文件")
        
        # 同步文件
        for local_path, rel_path in files_to_sync:
            if self._is_file_changed(local_path, rel_path):
                self._transfer_file(local_path, rel_path)
            else:
                self.stats.files_skipped += 1
                logger.debug(f"跳过（未修改）: {rel_path}")
        
        # 保存远程哈希缓存
        if not self.dry_run:
            self._save_remote_hashes()
    
    def _scan_directory(self, dir_path: Path, rel_prefix: str, 
                       files: List[Tuple[str, str]]):
        """
        递归扫描目录，收集文件列表
        
        Args:
            dir_path: 目录的绝对路径
            rel_prefix: 相对路径前缀
            files: 文件列表（输出参数）
        """
        try:
            for item in dir_path.iterdir():
                rel_path = f"{rel_prefix}/{item.name}"
                
                if item.is_dir():
                    # 检查目录是否应该被忽略
                    if self._should_ignore(rel_path, is_dir=True):
                        self.stats.files_ignored += 1
                        logger.debug(f"忽略目录: {rel_path}")
                        continue
                    
                    # 递归扫描子目录
                    self._scan_directory(item, rel_path, files)
                
                elif item.is_file():
                    # 检查文件是否应该被忽略
                    if self._should_ignore(rel_path, is_dir=False):
                        self.stats.files_ignored += 1
                        logger.debug(f"忽略文件: {rel_path}")
                        continue
                    
                    files.append((str(item), rel_path))
        except PermissionError as e:
            logger.warning(f"权限不足，无法访问 {dir_path}: {e}")
        except Exception as e:
            logger.error(f"扫描目录 {dir_path} 时出错: {e}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="LoongGuard 项目同步脚本")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际传输文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    parser.add_argument("--host", type=str, help="远程主机地址（覆盖配置）")
    parser.add_argument("--username", type=str, help="SSH 用户名（覆盖配置）")
    parser.add_argument("--password", type=str, help="SSH 密码（覆盖配置）")
    parser.add_argument("--remote-dir", type=str, help="远程目标目录（覆盖配置）")

    args = parser.parse_args()

    # 调整日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 允许通过命令行参数覆盖配置
    if args.host:
        SYNC_CONFIG["remote"]["host"] = args.host
    if args.username:
        SYNC_CONFIG["remote"]["username"] = args.username
    if args.password:
        SYNC_CONFIG["remote"]["password"] = args.password
    if args.remote_dir:
        SYNC_CONFIG["remote_dir"] = args.remote_dir
    
    # 创建同步器并执行
    syncer = LoongGuardSyncer(SYNC_CONFIG, dry_run=args.dry_run)

    try:
        # dry-run 模式仅扫描本地文件，无需 SSH 连接
        if not args.dry_run:
            if not syncer.connect():
                logger.error("无法建立连接，同步终止")
                sys.exit(1)

        syncer.scan_and_sync()

    except KeyboardInterrupt:
        logger.warning("用户中断同步")
    except Exception as e:
        logger.error(f"同步过程中发生未预期的错误: {e}")
    finally:
        syncer.disconnect()
        syncer.stats.print_summary()

    # 如果有错误，返回非零退出码
    if syncer.stats.errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
