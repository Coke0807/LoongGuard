import time
import signal
import sys
import pyaudio
import json
import re
import pyttsx3
from vosk import Model, KaldiRecognizer

# ====================== 全局业务配置 ======================
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024

WAKE_WORDS = {"小龙小龙", "晓珑监护"}
wake_active = False
last_wake_time = 0
WAKE_COOL = 3       # 唤醒冷却3秒
WAKE_TIMEOUT = 8    # 唤醒后8秒无指令自动待机

# 模式管理
CURRENT_MODE = "normal"  # normal: 正常课堂, nap: 午睡, public: 公共区域
MODE_NAMES = {
    "normal": "正常课堂",
    "nap": "午睡",
    "public": "公共区域"
}

# ====================== 语音播报函数（独立引擎，防无声） ======================
def speak(text):
    print(f"【语音播报】{text}")
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)
        engine.setProperty("volume", 0.95)
        # 尝试加载中文语音
        voices = engine.getProperty('voices')
        for voice in voices:
            if "zh" in voice.id or "Chinese" in voice.name:
                engine.setProperty('voice', voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"播报错误：{e}")
    finally:
        if engine is not None:
            try:
                engine.stop()
            except:
                pass
            del engine

# ====================== 语音指令处理逻辑 ======================
def handle_command(txt):
    """
    处理唤醒后的语音指令，播报对应反馈。
    返回 True 表示指令已被识别并处理，否则 False。
    """
    global wake_active, CURRENT_MODE

    # ====== 1. 退出程序（最高优先级） ======
    if "退出程序" in txt or "关闭程序" in txt or "退出" in txt:
        speak("AI监护已关闭")
        time.sleep(1)
        sys.exit(0)

    # ====== 2. 模式切换指令 ======
    if "切换到正常课堂模式" in txt:
        CURRENT_MODE = "normal"
        speak("已切换到正常课堂模式")
        return True
    if "切换到午睡模式" in txt:
        CURRENT_MODE = "nap"
        speak("已切换到午睡模式")
        return True
    if "切换到公共区域模式" in txt:
        CURRENT_MODE = "public"
        speak("已切换到公共区域模式")
        return True
    if "切换模式" in txt:
        # 模糊指令，没有指定具体模式
        speak("请说出你需要切换的模式")
        return True

    # ====== 3. 关闭告警 ======
    if "关闭告警" in txt or "关闭当前告警" in txt:
        speak("已关闭当前告警")
        # 此处可调用实际关闭告警的接口，暂留空
        return True

    # ====== 4. 查询温湿度（模拟数据） ======
    if "查询温湿度" in txt or "温湿度" in txt:
        # 模拟随机温湿度值
        import random
        temp = round(random.uniform(18.0, 28.0), 1)
        humi = round(random.uniform(40.0, 70.0), 1)
        speak(f"当前环境温度{temp}摄氏度，湿度{humi}%")
        return True

    # ====== 5. 开始录制 ======
    if "开始录制" in txt:
        speak("视频录制已开启")
        # 此处可调用实际录制接口
        return True

    # ====== 6. 查看日志 ======
    if "查看日志" in txt or "调取日志" in txt:
        speak("正在读取设备运行日志")
        # 此处可调取日志接口
        return True

    # ====== 7. 关闭识别（返回待机） ======
    if "关闭识别" in txt:
        wake_active = False
        speak("已回到待机，唤醒请说小龙小龙")
        return True

    # ====== 如果以上均未匹配，视为未识别指令 ======
    return False

# ====================== 麦克风收音主逻辑 ======================
def run_audio_task():
    global wake_active, last_wake_time, CURRENT_MODE
    # 加载离线语音识别模型
    model = Model("./vosk-model-cn-0.22")
    recog = KaldiRecognizer(model, SAMPLE_RATE)
    recog.SetWords(True)

    # 开启麦克风采集
    pya = pyaudio.PyAudio()
    stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # 开机播报
    speak("AI监护已启动")
    print("=== 语音监护系统启动 ===")
    print("唤醒词：小龙小龙、晓珑监护")
    print("当前模式：", MODE_NAMES.get(CURRENT_MODE, "正常课堂"))
    print("========================\n")

    wake_start_time = 0
    while True:
        data = stream.read(CHUNK)
        time.sleep(0.05)  # 让出CPU

        if recog.AcceptWaveform(data):
            res_json = json.loads(recog.Result())
            text_result = res_json.get("text", "").strip()
            if not text_result:
                continue
            print(f"识别文字：{text_result}")
            clean_text = re.sub(r"\s+", "", text_result)
            now_timestamp = time.time()

            # 待机状态：检测唤醒词
            if not wake_active:
                for wake_word in WAKE_WORDS:
                    if wake_word in clean_text and (now_timestamp - last_wake_time > WAKE_COOL):
                        wake_active = True
                        last_wake_time = now_timestamp
                        wake_start_time = now_timestamp
                        speak("我在")
                        break
            # 唤醒激活状态：解析指令
            else:
                handled = handle_command(clean_text)
                if not handled:
                    pass
                # 超时自动待机
                if now_timestamp - wake_start_time > WAKE_TIMEOUT:
                    wake_active = False
                    print("唤醒超时，自动返回待机")

# ====================== 捕获Ctrl+C安全退出 ======================
def exit_handler(sig_num, frame):
    print("\n程序收到退出信号，正在关闭...")
    try:
        speak("AI监护已关闭")
    except:
        pass
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_handler)
    try:
        run_audio_task()
    finally:
        print("程序正常结束")