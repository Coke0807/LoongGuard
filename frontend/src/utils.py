import os
from datetime import datetime, timedelta
import subprocess

# ===================== 日志全局常量 =====================
LOG_DIR = "./log"
VIDEO_SAVE_DIR = "./video_record"
CAPTURE_SAVE_DIR = "./capture"
RETENTION_DAY = 30  # 视频保留30天

# 初始化目录
for folder in [LOG_DIR, VIDEO_SAVE_DIR, CAPTURE_SAVE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_log_file_path():
    """获取当天日志文件路径"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(LOG_DIR, f"app_{today}.log")

def write_log(msg_type: str, msg: str):
    """
    写入日志，同时打印控制台
    :param msg_type: INFO/WARN/ERROR/CAM/NET/SENSOR/MIC/RECORD
    :param msg: 日志内容
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{now_str}] [{msg_type}] {msg}"
    print(log_line)
    log_path = get_log_file_path()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[LOG_ERROR] 写入日志失败: {str(e)}")

# ===================== 自动清理30天前视频 =====================
def clean_expired_video():
    cutoff = datetime.now() - timedelta(days=RETENTION_DAY)
    all_files = os.listdir(VIDEO_SAVE_DIR)
    del_count = 0
    for fname in all_files:
        if not fname.startswith("video_") or not fname.endswith(".mp4"):
            continue
        try:
            time_str = fname.replace("video_", "").replace(".mp4", "")
            file_dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            if file_dt < cutoff:
                os.remove(os.path.join(VIDEO_SAVE_DIR, fname))
                del_count += 1
        except Exception:
            continue
    if del_count > 0:
        write_log("RECORD", f"自动清理{RETENTION_DAY}天前视频，删除{del_count}个过期文件")


def record_audio_merge(video_path, fps):
    audio_path = video_path.replace(".mp4", ".wav")
    # 录制麦克风音频
    cmd_record_audio = [
        #"arecord", "-f", "s16_le", "-r", "44100", "-c", "2", audio_path
        "arecord", "-D", "hw:1,0", "-f", "s16_le", "-r", "44100", "-c", "2", audio_path
    ]
    audio_proc = subprocess.Popen(cmd_record_audio)
    # 等待视频录制结束信号省略，停止音频
    return audio_proc, audio_path


def merge_video_audio(video_path, audio_path, out_path):
    write_log("RECORD", f"开始合并{video_path}和{audio_path}至{out_path}文件")
    # 音视频合并
    cmd_merge = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        out_path
    ]
    subprocess.run(cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 安全删除文件，先判断是否存在
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(video_path):
        os.remove(video_path)