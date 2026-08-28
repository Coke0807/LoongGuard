# LoongArch 远程开发完整指南

## 方案对比

| 方案 | AI 辅助程度 | 开发效率 | 推荐度 |
|------|-----------|---------|--------|
| **VS Code Remote SSH** | ⭐⭐⭐⭐⭐（完全 AI 辅助） | ⭐⭐⭐⭐⭐ | ✅ 强烈推荐 |
| **SSH + 手动编辑** | ⭐⭐（复制粘贴） | ⭐⭐ | ⚠️ 备选方案 |
| **纯手敲** | ⭐（无辅助） | ⭐ | ❌ 不推荐 |

---

## 方案 1：VS Code Remote SSH（推荐）

### 优势
- ✅ **AI 完全可用**：在远程 Linux 上，GitHub Copilot 完全可用
- ✅ **无缝开发**：Windows 上编辑，Linux 上实时运行
- ✅ **智能提示**：完整的代码补全、错误检查
- ✅ **调试支持**：可以直接在远程调试代码

### 配置步骤

#### 步骤 1：填写 SSH 配置

编辑 `.env` 文件：

```bash
# ── SSH 远程连接（龙芯开发板）────────────────────────────
LG_SSH_HOST=192.168.1.100      # ← 填写你的 LoongArch 设备 IP
LG_SSH_PORT=22
LG_SSH_USER=cc                 # ← SSH 用户名
LG_SSH_PASS=your_password      # ← SSH 密码
```

#### 步骤 2：配置 VS Code SSH

1. 按 `F1` 或 `Ctrl+Shift+P`
2. 输入 `Remote-SSH: Open SSH Configuration File`
3. 选择第一个配置文件（通常是 `C:\Users\你的用户名\.ssh\config`）
4. 添加以下内容：

```
Host loongarch
    HostName 192.168.1.100      # ← 替换为你的设备 IP
    User cc                     # ← 替换为你的用户名
    Port 22
```

#### 步骤 3：连接到远程设备

1. 按 `F1`
2. 输入 `Remote-SSH: Connect to Host`
3. 选择 `loongarch`
4. 输入密码（首次连接需要确认指纹）

#### 步骤 4：在远程打开项目

连接成功后：
1. `File` → `Open Folder`
2. 选择 `/home/cc/project/loong-guard`
3. 现在你可以在 Windows 上编辑，代码在 Linux 上运行！

#### 步骤 5：安装远程扩展

在远程环境中安装：
- Python 扩展
- Pylance
- GitHub Copilot（如果可用）

### 使用 AI 辅助开发

连接到远程后，你可以：

```python
# 在远程 Linux 上创建 v4l2_ext.c

// 1. 在 VS Code 中新建文件
// 2. 输入文件名：src/camera/v4l2_ext.c
// 3. 使用 AI 生成代码：

/*
 * V4L2 摄像头采集 C 扩展
 * 
 * 功能：直接通过 V4L2 ioctl 操作摄像头
 * 目标平台：LoongArch64 + Loongnix_v25
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

// ... AI 会帮你生成完整的实现
```

### 编译和测试

在 VS Code 终端中（已连接到远程）：

```bash
# 编译 C 扩展
gcc -shared -fPIC -o src/camera/v4l2_ext.so src/camera/v4l2_ext.c

# 测试
python -m src.pipeline
```

---

## 方案 2：SSH + 本地编辑（备选）

如果 VS Code Remote 不可用，可以使用这个方案：

### 工作流程

```
1. Windows 上用 AI 生成代码
   ↓
2. 保存到本地文件
   ↓
3. 使用脚本自动同步到 Linux
   ↓
4. SSH 登录 Linux 编译测试
```

### 配置自动同步

#### 步骤 1：配置 SSH 密钥（避免每次输入密码）

```powershell
# Windows 上生成密钥
ssh-keygen -t rsa -b 4096

# 复制公钥到 LoongArch
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh cc@192.168.1.100 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 测试免密登录
ssh cc@192.168.1.100
```

#### 步骤 2：创建同步脚本

创建 `scripts/sync_to_loongarch.ps1`：

```powershell
#!/usr/bin/env pwsh
param(
    [string]$RemoteHost = "192.168.1.100",
    [string]$RemoteUser = "cc",
    [string]$RemotePath = "/home/cc/project/loong-guard"
)

$LocalPath = "D:\User\Desktop\LoongGuard"

Write-Host "同步代码到 LoongArch..." -ForegroundColor Green

# 使用 rsync 或 scp 同步
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' `
    "$LocalPath/" "$RemoteUser@$RemoteHost:$RemotePath/"

Write-Host "同步完成！" -ForegroundColor Green
```

#### 步骤 3：使用 AI 辅助开发

```powershell
# 1. 在 Windows 上用 AI 生成代码
# 2. 保存文件
# 3. 运行同步脚本
.\scripts\sync_to_loongarch.ps1

# 4. SSH 登录测试
ssh cc@192.168.1.100
cd /home/cc/project/loong-guard
python -m src.pipeline
```

---

## 方案 3：使用 AI 生成完整实现

### 步骤 1：让 AI 生成 V4L2 C 扩展代码

在对话中告诉我：

```
请为 LoongArch 生成完整的 v4l2_ext.c 实现，包括：
1. v4l2_open：打开设备并初始化
2. v4l2_read_frame：读取帧数据
3. v4l2_close：关闭设备
4. 支持 MJPEG 和 YUYV 格式
5. 使用 mmap 零拷贝
```

### 步骤 2：AI 生成代码

我会生成完整的 C 代码，你只需要：
1. 复制代码
2. 保存到 `src/camera/v4l2_ext.c`
3. 同步到 Linux
4. 编译测试

### 步骤 3：自动化测试

创建测试脚本 `scripts/test_v4l2.sh`：

```bash
#!/bin/bash
# 在 LoongArch 上运行

echo "编译 V4L2 扩展..."
gcc -shared -fPIC -o src/camera/v4l2_ext.so src/camera/v4l2_ext.c

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "测试运行..."
    python -m src.pipeline
else
    echo "编译失败！"
    exit 1
fi
```

---

## 实际开发示例

### 示例 1：使用 VS Code Remote 开发 V4L2 扩展

```
1. 连接到远程 LoongArch
   ↓
2. 创建 src/camera/v4l2_ext.c
   ↓
3. 输入注释，AI 自动补全：
   /* v4l2_open - 打开 V4L2 设备
    * 
    * 参数：
    *   device - 设备路径，如 "/dev/video0"
    *   width  - 期望宽度
    *   height - 期望高度
    * 
    * 返回：
    *   文件描述符（>=0），失败返回 -1
    */
   int v4l2_open(const char *device, int width, int height, int pixel_format) {
       // AI 会自动生成实现...
   }
   ↓
4. 在远程终端编译：
   gcc -shared -fPIC -o v4l2_ext.so v4l2_ext.c
   ↓
5. 测试：
   python -m src.pipeline
```

### 示例 2：使用 AI 生成完整模块

```
你：请生成一个完整的 V4L2 C 扩展，支持以下功能：
    - 打开摄像头设备
    - 设置分辨率和格式
    - 使用 mmap 读取帧数据
    - 支持 MJPEG 和 YUYV 格式
    - 错误处理和日志输出

AI：[生成完整的 v4l2_ext.c 代码]

你：[复制代码到文件]
   [同步到 Linux]
   [编译测试]
```

---

## 常见问题

### Q1: VS Code Remote 连接失败？

**解决方案**：
```bash
# 检查 SSH 服务是否运行
ssh cc@192.168.1.100

# 检查防火墙
sudo ufw status
sudo ufw allow 22

# 检查 SSH 配置
sudo systemctl status sshd
```

### Q2: AI 在远程环境不可用？

**解决方案**：
- GitHub Copilot 需要在远程环境重新登录
- 或使用本地 AI 生成代码，然后同步到远程

### Q3: 编译失败？

**解决方案**：
```bash
# 安装依赖
sudo apt-get install build-essential linux-headers-$(uname -r)

# 检查头文件
ls /usr/include/linux/videodev2.h
```

---

## 推荐工作流程

### 最优方案：VS Code Remote SSH

```
Windows (VS Code) ←SSH→ LoongArch (运行代码)
     ↓
  AI 辅助
     ↓
  实时编辑、编译、测试
```

**优势**：
- ✅ AI 完全可用
- ✅ 无需手动同步
- ✅ 实时反馈
- ✅ 完整的开发体验

### 备选方案：本地编辑 + 自动同步

```
Windows (编辑) → 同步脚本 → LoongArch (编译测试)
     ↓
  AI 辅助
     ↓
  一键同步
```

**优势**：
- ✅ AI 可用
- ✅ 简单易用
- ⚠️ 需要手动同步

---

## 下一步

1. **配置 SSH 连接**：填写 `.env` 文件中的 SSH 配置
2. **选择开发方案**：推荐 VS Code Remote SSH
3. **开始开发**：使用 AI 辅助生成 V4L2 代码

需要我帮你：
- 生成完整的 V4L2 C 扩展代码？
- 配置 VS Code Remote SSH？
- 创建自动同步脚本？
