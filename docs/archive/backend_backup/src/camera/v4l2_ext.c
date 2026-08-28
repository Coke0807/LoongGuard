/*
 * v4l2_ext.c -- V4L2 摄像头采集 C 扩展（阶段四框架代码）
 *
 * 设计动机：
 *     直接通过 V4L2 ioctl 操作摄像头，使用 mmap 零拷贝 DMA 获取帧数据，
 *     避免 OpenCV VideoCapture 的额外内存拷贝开销。
 *     目标平台：龙芯 2K3000 (LoongArch64) + Loongnix_v25
 *
 * 编译：
 *     gcc -shared -fPIC -o v4l2_ext.so v4l2_ext.c
 *     # 或使用同目录下的 Makefile: make
 *
 * Python 调用：
 *     import ctypes
 *     lib = ctypes.CDLL("./v4l2_ext.so")
 *     fd = lib.v4l2_open(b"/dev/video0", 640, 480, 30, 0)  # 0=YUYV
 *     buf_size = lib.v4l2_get_frame_size(fd)
 *     buf = (ctypes.c_uint8 * buf_size)()
 *     while True:
 *         ret = lib.v4l2_read_frame(fd, buf, buf_size)
 *         if ret > 0:
 *             # buf 中为原始像素数据，转为 numpy 使用
 *             frame = np.frombuffer(buf, dtype=np.uint8).reshape(480, 640, 2)
 *     lib.v4l2_close(fd)
 *
 * TODO（阶段四实现）：
 *     [ ] 实现 v4l2_open 中的 VIDIOC_QUERYCAP / VIDIOC_S_FMT / VIDIOC_REQBUFS
 *     [ ] 实现 mmap 缓冲区映射
 *     [ ] 实现 v4l2_read_frame 中的 VIDIOC_DQBUF / VIDIOC_QBUF 循环
 *     [ ] 添加 YUYV -> RGB 转换（或在 Python 侧用 cv2.cvtColor）
 *     [ ] 添加 MJPEG 解码支持
 *     [ ] 在 LoongArch 实机上验证 DMA 零拷贝性能
 *     [ ] 添加错误码定义和详细错误信息
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

/* 缓冲区数量（环形缓冲） */
#define NUM_BUFFERS 4

/* 帧缓冲区描述 */
struct buffer {
    void   *start;
    size_t  length;
};

/* 设备上下文 */
struct v4l2_ctx {
    int              fd;
    uint32_t         width;
    uint32_t         height;
    uint32_t         fps;
    uint32_t         pixfmt;       /* V4L2_PIX_FMT_YUYV 等 */
    size_t           frame_size;   /* 单帧字节数 */
    struct buffer    buffers[NUM_BUFFERS];
    uint32_t         n_buffers;
    int              is_streaming;
};

/* 全局上下文（简化实现，生产环境应支持多设备） */
static struct v4l2_ctx g_ctx __attribute__((unused)) = { .fd = -1 };


/**
 * v4l2_open -- 打开 V4L2 设备并初始化采集
 *
 * @param device       设备路径，如 "/dev/video0"
 * @param width        期望宽度
 * @param height       期望高度
 * @param pixel_format V4L2 像素格式常量（0x3352424=BGR24, 0x47504A4D=MJPG）
 * @return             文件描述符 fd（>=0），失败返回 -1
 */
int v4l2_open(const char *device, int width, int height,
              int pixel_format)
{
    /* TODO: 阶段四实现 */
    fprintf(stderr, "[v4l2_ext] v4l2_open: NOT IMPLEMENTED (stub)\n");
    (void)device; (void)width; (void)height; (void)pixel_format;
    return -1;

    /*
     * 实际实现步骤：
     *
     * 1. fd = open(device, O_RDWR | O_NONBLOCK)
     *
     * 2. struct v4l2_capability cap;
     *    ioctl(fd, VIDIOC_QUERYCAP, &cap)
     *    检查 cap.capabilities & V4L2_CAP_VIDEO_CAPTURE
     *    检查 cap.capabilities & V4L2_CAP_STREAMING
     *
     * 3. struct v4l2_format fmt;
     *    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     *    fmt.fmt.pix.width = width;
     *    fmt.fmt.pix.height = height;
     *    fmt.fmt.pix.pixelformat = pixel_format;
     *    fmt.fmt.pix.field = V4L2_FIELD_NONE;
     *    ioctl(fd, VIDIOC_S_FMT, &fmt)
     *    // 实际协商后的分辨率可能与请求不同，需更新 ctx
     *
     * 4. struct v4l2_requestbuffers req;
     *    req.count = NUM_BUFFERS;
     *    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     *    req.memory = V4L2_MEMORY_MMAP;
     *    ioctl(fd, VIDIOC_REQBUFS, &req)
     *
     * 5. 对每个 buffer:
     *    ioctl(fd, VIDIOC_QUERYBUF, &buf)
     *    buffers[i].start = mmap(NULL, buf.length, PROT_READ|PROT_WRITE,
     *                            MAP_SHARED, fd, buf.m.offset)
     *    ioctl(fd, VIDIOC_QBUF, &buf)  // 归还缓冲区给驱动
     *
     * 6. enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     *    ioctl(fd, VIDIOC_STREAMON, &type)
     *
     * 7. 保存所有状态到 g_ctx，返回 fd
     */
}


/**
 * v4l2_read_frame -- 读取一帧原始数据
 *
 * @param fd        v4l2_open 返回的文件描述符
 * @param buf       用户提供的输出缓冲区（BGR24 数据）
 * @param buf_size  缓冲区大小
 * @param out_width 输出参数，实际帧宽度（可为 NULL）
 * @param out_height 输出参数，实际帧高度（可为 NULL）
 * @return          实际读取的字节数，0=无帧可用，-1=错误
 */
int v4l2_read_frame(int fd, uint8_t *buf, size_t buf_size,
                    int *out_width, int *out_height)
{
    /* TODO: 阶段四实现 */
    (void)fd; (void)buf; (void)buf_size;
    if (out_width)  *out_width = 0;
    if (out_height) *out_height = 0;
    return -1;

    /*
     * 实际实现步骤：
     *
     * 1. struct v4l2_buffer vbuf;
     *    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     *    vbuf.memory = V4L2_MEMORY_MMAP;
     *    ioctl(fd, VIDIOC_DQBUF, &vbuf)  // 从驱动取出一个已填充的缓冲区
     *
     * 2. size_t copy_len = min(vbuf.bytesused, buf_size);
     *    memcpy(buf, buffers[vbuf.index].start, copy_len);
     *
     * 3. ioctl(fd, VIDIOC_QBUF, &vbuf)  // 归还缓冲区给驱动
     *
     * 4. return copy_len;
     *
     * 注意：如果像素格式为 MJPEG，buf 中的数据需要额外解码
     * 可选择在 C 侧用 libjpeg-turbo 解码，或在 Python 侧用 cv2.imdecode
     */
}


/**
 * v4l2_get_frame_size -- 获取单帧字节数
 *
 * @param fd  文件描述符
 * @return    单帧字节数，-1 表示错误
 */
int v4l2_get_frame_size(int fd)
{
    (void)fd;
    /* TODO: 返回 g_ctx.frame_size（由 VIDIOC_S_FMT 协商确定） */
    return -1;
}


/**
 * v4l2_get_resolution -- 获取设备协商后的实际分辨率
 *
 * @param fd         v4l2_open 返回的文件描述符
 * @param out_width  输出参数，实际宽度
 * @param out_height 输出参数，实际高度
 * @return           0=成功，-1=错误
 */
int v4l2_get_resolution(int fd, int *out_width, int *out_height)
{
    /* TODO: 从 g_ctx 中读取 VIDIOC_S_FMT 协商后的实际分辨率 */
    (void)fd;
    if (out_width)  *out_width = 0;
    if (out_height) *out_height = 0;
    return -1;
}


/**
 * v4l2_close -- 停止采集并释放所有资源
 *
 * @param fd  文件描述符
 */
void v4l2_close(int fd)
{
    /* TODO: 阶段四实现 */
    (void)fd;

    /*
     * 实际实现步骤：
     *
     * 1. enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     *    ioctl(fd, VIDIOC_STREAMOFF, &type)
     *
     * 2. for (i = 0; i < n_buffers; i++)
     *        munmap(buffers[i].start, buffers[i].length)
     *
     * 3. close(fd)
     * 4. memset(&g_ctx, 0, sizeof(g_ctx))
     */
}
