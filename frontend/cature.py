import cv2

# 0对应/dev/video0，1对应video1
cap = cv2.VideoCapture(0)
# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("摄像头打开失败，请检查/dev/video0权限")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Loongnix Camera", frame)
        # ESC退出
        if cv2.waitKey(20) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
