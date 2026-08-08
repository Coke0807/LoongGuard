from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# ===================== 告警弹窗 =====================
class WarnDialog(QDialog):
    def __init__(self, warn_type, warn_msg):
        super().__init__()
        self.setWindowTitle("⚠️ 园区安全预警")
        self.setFixedSize(440, 240)
        self.setStyleSheet("background-color: #fff3f3; border-radius:12px;")
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(12)
        title = QLabel(f"【{warn_type}】")
        title.setFont(QFont("Noto Sans SC", 15, QFont.Weight.Bold))
        title.setStyleSheet("color:#d92121;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = QLabel(warn_msg)
        msg.setFont(QFont("Noto Sans SC",12))
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_ok = QPushButton("确认处理")
        btn_ok.setFixedHeight(40)
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.setStyleSheet("""
            QPushButton{background:#3488d8;color:white;border-radius:8px;font-size:11pt;}
            QPushButton:hover{background:#2770b8;}
        """)
        btn_ok.clicked.connect(self.close)
        layout.addWidget(title)
        layout.addWidget(msg)
        layout.addWidget(btn_ok)
        self.setLayout(layout)