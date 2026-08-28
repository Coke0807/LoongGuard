# V4L2 C 扩展编译 -- 仅在 LoongArch Linux 目标板上使用
# 用法: make          编译共享库
#       make clean    清理构建产物
#
# 前置条件: gcc, linux-headers (V4L2 头文件)

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -fPIC
TARGET  = v4l2_ext.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): v4l2_ext.c
	$(CC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f $(TARGET)
