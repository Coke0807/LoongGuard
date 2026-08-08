# Docker 运行 FunASR（龙芯平台）

> 在龙芯 Debian 容器中源码编译 Python 3.10、ONNX Runtime，并安装部署 FunASR 语音识别模型。

## 一、启动容器

```bash
docker run -dit cr.loongnix.cn/library/debian:buster /bin/bash
```

## 二、容器内操作步骤

### 1. 更新源、安装编译全套依赖

```bash
apt update
apt install -y build-essential cmake git wget curl libssl-dev \
libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
libopenblas-dev ffmpeg pkg-config
```

### 2. 源码编译 Python 3.10.15（buster 无官方高版本 python）

```bash
cd /usr/src
wget https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz
tar -zxvf Python-3.10.15.tgz
cd Python-3.10.15

# loongarch 编译配置
./configure --prefix=/opt/python3.10 --enable-shared
make -j$(nproc)
make install

# 动态链接
echo "/opt/python3.10/lib" > /etc/ld.so.conf.d/python310.conf
ldconfig

# 创建软链接
ln -s /opt/python3.10/bin/python3.10 /usr/local/bin/python3.10
ln -s /opt/python3.10/bin/pip3.10 /usr/local/bin/pip3.10
```

验证：

```bash
python3.10 -V
pip3.10 -V
```

### 3. 创建虚拟环境（隔离）

```bash
python3.10 -m venv /app/venv
source /app/venv/bin/activate

pip install --upgrade pip setuptools wheel
python3.10 -m pip install packaging numpy
```

### 4. 依赖关键点（LoongArch 无预编译包）

> ⚠️ **不能直接 `pip install onnxruntime`**，没有 loongarch64 包！

#### 方式 1：源码编译 ONNX Runtime（推荐）

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

> 注意事项：
> - 镜像自带 python2.7，需更改环境变量，将 python3 版本切换为 python3.10。
> - cmake 版本过低，需源码编译升级到 **3.28**。

#### 升级 CMake 到 3.28（龙芯 2K3000 编译约 15 分钟）

```bash
# 1. 卸载旧版本CMake（可选）
apt remove cmake -y

# 2. 安装依赖
apt update
apt install -y wget build-essential libssl-dev

# 3. 下载并安装CMake 3.28+
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0.tar.gz
tar -xzvf cmake-3.28.0.tar.gz
cd cmake-3.28.0

# 4. 编译&安装
./bootstrap --prefix=/usr/local   # loongarch 2k3000 大约10分钟
make -j$(nproc)                  # 内存不足可改为 make -j1；loongarch 2k3000 需5分钟
```

编译到 `[100%] Built target CMakeLibTests` 后执行安装：

```bash
sudo make install
```

验证并设置环境变量：

```bash
/usr/local/bin/cmake --version
cmake version 3.28.0

echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 编译 ONNX Runtime

```bash
./build.sh --config Release --build_shared_lib --skip_tests --allow_running_as_root
```

编译完成后在 `build/Linux/Release` 生成 whl：

```bash
pip install ./build/Linux/Release/dist/*.whl
```

### 5. 安装 FunASR & ModelScope

```bash
pip install modelscope
pip install funasr
```

### 6. 最简测试代码 test_asr.py

```python
from funasr import AutoModel

model = AutoModel(
    model="speech_paraformer_asr_nat-zh-cn-16k-nano",
    device="cpu",
    backend="onnxruntime"
)

res = model.generate(input="test.wav")
print(res)
```

运行测试：

```bash
python test_asr.py
```