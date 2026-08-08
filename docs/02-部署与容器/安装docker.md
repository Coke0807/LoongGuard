# 安装 Docker（龙芯平台）

> 在 Loongnix 龙芯平台上安装 Docker 并配置龙芯镜像加速。

## 1. 安装 Docker

```bash
sudo apt update
sudo apt install docker.io
```

## 2. 配置镜像加速（重要）

编辑或创建 `/etc/docker/daemon.json` 文件，添加龙芯的官方镜像仓库，拉取镜像会快很多：

```json
{
  "registry-mirrors": ["https://cr.loongnix.cn"]
}
```

然后重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 3. 拉取镜像测试

用下面的命令拉取一个龙芯的 Debian 基础镜像，测试是否配置成功：

```bash
sudo docker pull cr.loongnix.cn/library/debian:buster
sudo docker run -it cr.loongnix.cn/library/debian:buster /bin/bash
```

验证结果：

```text
root@tjdz-pc:~# docker images
REPOSITORY                TAG     IMAGE ID       CREATED      SIZE
cr.loongnix.cn/library/debian  buster  abcf73d8d8ba  8 months ago  120MB
```