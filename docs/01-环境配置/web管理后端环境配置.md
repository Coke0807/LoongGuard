# Web 管理后端环境配置

> 基于 **FastAPI** 的 web 管理后台环境搭建（Loongnix 龙芯平台）。

## 1. 安装 FastAPI 框架依赖

进入虚拟环境：

```bash
tjdz@tjdz-pc:~/venv-ui$ source bin/activate
(venv-ui) tjdz@tjdz-pc:~/venv-ui$ pwd
/home/tjdz/venv-ui
```

安装 FastAPI：

```bash
(venv-ui) tjdz@tjdz-pc:~/venv-ui$ pip install fastapi
```

已安装版本：

| 包 | 版本 |
| --- | --- |
| fastapi | 0.139.0 |
| starlette | 1.3.1 |
| pydantic | 2.13.4 |
| pydantic-core | 2.46.4 |
| typing-extensions | 4.16.0 |
| anyio | 4.14.2 |

> 源地址：`https://lpypi.loongnix.cn/loongson/pypi`、`https://pypi.tuna.tsinghua.edu.cn/simple`

## 2. 安装 openpyxl 3.1.5

> ⚠️ 使用**华为源**，清华和龙芯源版本都较低。

```bash
(venv-ui) tjdz@tjdz-pc:~/venv-ui$ pip install openpyxl==3.1.5 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

## 3. 安装 python-multipart 0.0.32

```bash
(venv-ui) tjdz@tjdz-pc:~/vss/website$ pip install python-multipart -i  https://repo.huaweicloud.com/repository/pypi/simple/
```

## 4. 安装 jinja2 3.1.6

```bash
(venv-ui) tjdz@tjdz-pc:~/vss/website$ pip install jinja2 -i  https://repo.huaweicloud.com/repository/pypi/simple/
```

> MarkupSafe 在龙芯上编译产出的 wheel：`markupsafe-3.0.3-cp313-cp313-linux_loongarch64.whl`

## 5. 启动管理后台

```bash
(venv-ui) tjdz@tjdz-pc:~/vss/website$ sudo $(which python3) main.py
INFO:     Started server process [4106]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on https://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 6. 配置防火墙

查看防火墙状态：

```bash
tjdz@tjdz-pc:~$ systemctl status firewalld
```

禁用防火墙：

```bash
tjdz@tjdz-pc:~$ systemctl disable firewalld --now
Removed '/etc/systemd/system/multi-user.target.wants/firewalld.service'.
Removed '/etc/systemd/system/dbus-org.fedoraproject.FirewallD1.service'.
```

## 浏览器访问

```text
http://127.0.0.1:8000/admin/login
```