# UI 端开发环境配置

> 适用于 **Loongnix 25.1** 龙芯平台，配置 PyQt6 前后端 UI 开发环境。

> ⚠️ **2026-08-26 更新**：本文记录的是早期 UI 开发环境搭建过程（当时的项目结构为
> `/home/tjdz/vss` + `ui.py`）。当前项目结构已迭代为 `frontend/` + `run.py`，
> **正式部署请以《[02-部署与容器/龙芯2K3000手动部署文档.md](../02-部署与容器/龙芯2K3000手动部署文档.md)》为准**。
> 本文仍有参考价值的部分：PyQt6 的 apt 安装方式、`QT_QPA_PLATFORM=xcb` 报错解决、
> `--system-site-packages` 虚拟环境思路（均已并入部署文档第 7 节）。

## 环境信息

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Loongnix 25.1 |
| Python | 3.13.5 |
| PyQt | 6.9.0 |
| 记录时间 | 2026-06-17 |
| 作者 | Eric.cao |

> ⚠️ 管理员密码：`loong123`

---

## 1. 安装系统依赖包

```bash
sudo apt install python3-pyqt6
```

## 2. 查看已安装依赖

```bash
sudo pip list
```

| Package | Version |
| --- | --- |
| arrow | 1.3.0 |
| attrs | 25.3.0 |
| bcrypt | 4.2.0 |
| cryptography | 43.0.0 |
| dbus-python | 1.4.0 |
| distro | 1.9.0 |
| fqdn | 1.5.1 |
| idna | 3.10 |
| ... | ... |
| PyQt6 | 6.9.0 |
| PyQt6_sip | 13.10.0 |
| tornado | 6.4.2 |

---

## 3. 创建虚拟环境（允许访问系统包）

```bash
python3 -m venv venv-ui --system-site-packages
```

激活虚拟环境：

```bash
source venv-ui/bin/activate
(venv-ui) tjdz@tjdz-pc:~$
```

---

## 4. 服务开机自启动配置

### 4.1 创建 autostart 目录

```bash
mkdir -p ~/.config/autostart
```

### 4.2 编写 desktop 启动文件

```bash
vim ~/.config/autostart/ai_guard_autostart.desktop
```

```ini
[Desktop Entry]
Type=Application
Name=幼儿园AI安全卫士
Comment=龙芯幼儿园AI安全卫士界面
Exec=/home/tjdz/ui_boot.sh
Terminal=false
StartupNotify=false
# 当前桌面是KDE
# 查看桌面是KDE还是XFCE echo $XDG_CURRENT_DESKTOP
OnlyShowIn=KDE
Hidden=false
```

**说明：**

- `X-Xfce-Delay=2`：延迟 2 秒启动，等待显卡、摄像头驱动加载完毕，避免 Qt 启动报错黑屏。
- `env QT_QPA_PLATFORM=xcb`：解决 Loongnix 下 PyQt6 报 **xcb 缺失**无法打开窗口的问题。

### 4.3 设置权限

```bash
chmod 644 ~/.config/autostart/ai_guard_autostart.desktop
```

### 4.4 编写启动脚本

```bash
vim /tjdz/home/ui_boot.sh
```

```bash
#!/bin/bash
# 日志路径，开机失败全部打印在这里
LOG=/home/tjdz/ui_boot_log.txt
echo "==================== $(date) 开机自启动开始 ====================" >> $LOG

# 加长等待，给显卡、Xorg、摄像头驱动足够加载时间
sleep 6

# 补齐Qt、显示全套环境变量
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0.0
export XDG_SESSION_TYPE=x11
export QT_AUTO_SCREEN_SCALE_FACTOR=1

# 进入代码目录，避免相对路径问题
cd /home/tjdz/vss

# 执行程序，stdout/stderr全部写入日志
/home/tjdz/venv-ui/bin/python3.13 ui.py >> $LOG 2>&1

# 程序退出打印标记，判断是否崩溃退出
echo "==================== $(date) 程序已退出 ====================" >> $LOG
echo "" >> $LOG
```

```bash
chmod +x /tjdz/home/ui_boot.sh
```

---

## 5. 摄像头部分

```bash
sudo apt update
sudo apt install python3-pyqt6 python3-opencv v4l-utils
```

摄像头权限：

```bash
sudo usermod -aG video $USER
```