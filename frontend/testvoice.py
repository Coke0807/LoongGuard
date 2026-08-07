import pyttsx3
eng = pyttsx3.init()
eng.say("测试语音，无网络、模式切换、传感器离线")
eng.runAndWait()
print("end")
