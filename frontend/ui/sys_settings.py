from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QCheckBox, QSpinBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QMouseEvent
from ui.settings_db import SettingsDB

from ui.voiceplayer import VoicePlayer

# 可点击主题卡片（修复__init参数顺序BUG）
class ThemeCardFrame(QFrame):
    def __init__(self, theme_name, main_dialog, parent=None):
        super().__init__(parent)
        self.theme_name = theme_name
        self.main_dialog = main_dialog
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.main_dialog.on_card_click(self.theme_name)
        super().mouseReleaseEvent(event)

class SysSettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.resize(960,640)
        # 全局样式：暖橙色底色，移除自定义控件QSS，修复Group重叠
        self.setStyleSheet("""
        QDialog{background: #fff0d8;}
        QTabWidget::pane {border-radius:12px;background:#ffffff;}
        QTabBar::tab{
            background:#ffddb0;min-width:90px;padding:8px 12px;
            border-top-left-radius:10px;border-top-right-radius:10px;
            font-size:11pt;font-family:"Noto Sans SC";
        }
        QTabBar::tab:selected{background:#ffb370;font-weight:bold;}
        QGroupBox{
            background:#fffaf0;border-radius:10px;border:1px solid #ffcc99;
            margin:12px 4px 4px 4px;
            font-size:11pt;font-weight:bold;color:#994400;
        }
        QGroupBox::title{
            subcontrol-origin:margin;
            subcontrol-position:top left;
            padding:0px 6px;
            margin-top:-8px;
            background:#fff0d8;
        }
        QLineEdit,QSpinBox{
            border:1px solid #ffc290;border-radius:6px;padding:5px;background:#fff;
        }
        QPushButton{
            border-radius:8px;padding:6px 14px;font-size:10pt;font-family:"Noto Sans SC";
        }
        QPushButton#saveBtn{background:#73d160;color:white;border:none;}
        QPushButton#saveBtn:hover{background:#5eb84b;}
        QPushButton#closeBtn{background:#ff9480;color:white;border:none;}
        QPushButton#closeBtn:hover{background:#e87c68;}
        QPushButton#browseBtn{background:#ffc470;color:#333;border:none;}
        QTableWidget{border-radius:8px;background:#fff;gridline-color:#ffe0c0;}
        QHeaderView::section{background:#ffddb0;padding:4px;border:none;}
        QCheckBox{font-size:10.5pt;}
        QLabel{font-family:"Noto Sans SC";font-size:10.5pt;}
    """)
        self.db = SettingsDB()
        self.cur_selected_theme = ""
        self.theme_widget_map = {}
        self.init_ui()
        self.load_all_config()

        # 新建语音播放器实例
        self.voice_player = VoicePlayer()


    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12,12,12,12)
        main_layout.setSpacing(12)
        self.tab = QTabWidget()
        self.tab.setFont(QFont("Noto Sans SC", 10))
        self.tab.addTab(self.tab_wake(), "唤醒词配置")
        self.tab.addTab(self.tab_alarm(), "告警策略")
        self.tab.addTab(self.tab_record(), "视频录制")
        self.tab.addTab(self.tab_user(), "用户权限")
        self.tab.addTab(self.tab_ota(), "在线升级")
        self.tab.addTab(self.tab_ntp(), "NTP校时")
        self.tab.addTab(self.tab_theme(), "卡通主题")
        main_layout.addWidget(self.tab)
        self.tab.currentChanged.connect(self.on_tab_switch)


        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_save = QPushButton("保存全部配置")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setObjectName("saveBtn")
        self.btn_save.clicked.connect(self.save_all)
        self.btn_cancel = QPushButton("关闭")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setObjectName("closeBtn")
        self.btn_cancel.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    # Tab切换触发音效
    def on_tab_switch(self, index):
        # index顺序和addTab一一对应
        if index == 0:
            # 唤醒词配置
            self.voice_player.voice_word_settings()
        elif index == 1:
            # 告警策略
            self.voice_player.voice_alarm_policy()
        elif index == 2:
            # 视频录制
            self.voice_player.voice_video_record()
        elif index == 3:
            # 用户权限
            #self.voice_player.voice_tab_user()
            index=3
        elif index == 4:
            # 在线升级
            self.voice_player.voice_online_upgrade()
        elif index == 5:
            # NTP校时
            self.voice_player.voice_ntp_valid()
        elif index == 6:
            # 卡通主题
            self.voice_player.voice_cartoon_theme()


    # 1 唤醒词页面
    def tab_wake(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("语音唤醒词设置（支持1~3个自定义词）")
        form_lay = QFormLayout(group)
        form_lay.setContentsMargins(12,16,12,12)
        form_lay.setSpacing(14)
        self.ed_w1 = QLineEdit()
        self.ed_w1.setPlaceholderText("必填，例如：小卫士")
        self.ed_w2 = QLineEdit()
        self.ed_w2.setPlaceholderText("可选，无则留空")
        self.ed_w3 = QLineEdit()
        self.ed_w3.setPlaceholderText("可选，无则留空")
        form_lay.addRow("唤醒词1：", self.ed_w1)
        form_lay.addRow("唤醒词2：", self.ed_w2)
        form_lay.addRow("唤醒词3：", self.ed_w3)
        tip = QLabel("多个唤醒词可分开识别，适配不同园所称呼习惯")
        tip.setStyleSheet("color:#777;font-size:9.5pt;")
        form_lay.addRow("", tip)
        lay.addWidget(group)
        lay.addStretch()
        return w

    # 2 告警策略
    def tab_alarm(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("各类告警独立策略配置（三种运行模式分开控制）")
        g_lay = QVBoxLayout(group)
        g_lay.setContentsMargins(12,16,12,12)
        self.alarm_table = QTableWidget()
        self.alarm_table.setColumnCount(7)
        self.alarm_table.setHorizontalHeaderLabels(["告警类型","运行模式","启用","级别","发声","推送","录制"])
        self.alarm_table.verticalHeader().setDefaultSectionSize(26)
        g_lay.addWidget(self.alarm_table)
        lay.addWidget(group)
        lay.addStretch()
        return w

    # 3 录制配置
    def tab_record(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("全局视频录制参数")
        form_lay = QFormLayout(group)
        form_lay.setContentsMargins(12,16,12,12)
        form_lay.setSpacing(12)
        self.rec_switch = QCheckBox("全局开启视频录制")
        self.rec_w = QSpinBox()
        self.rec_w.setRange(640,2560)
        self.rec_h = QSpinBox()
        self.rec_h.setRange(480,1440)
        self.rec_days = QSpinBox()
        self.rec_days.setRange(7,365)
        self.rec_path = QLineEdit()
        self.rec_path.setPlaceholderText("视频保存文件夹路径")
        self.btn_browse = QPushButton("选择目录")
        self.btn_browse.setObjectName("browseBtn")
        self.btn_browse.clicked.connect(self.select_rec_path)
        path_box = QHBoxLayout()
        path_box.setSpacing(6)
        path_box.addWidget(self.rec_path)
        path_box.addWidget(self.btn_browse)
        self.rec_pre = QCheckBox("开启告警事件预录")
        form_lay.addRow("", self.rec_switch)
        form_lay.addRow("画面宽度：", self.rec_w)
        form_lay.addRow("画面高度：", self.rec_h)
        form_lay.addRow("文件自动保留天数：", self.rec_days)
        form_lay.addRow("存储目录：", path_box)
        form_lay.addRow("", self.rec_pre)
        lay.addWidget(group)
        lay.addStretch()
        return w

    def select_rec_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择视频存储目录")
        if path:
            self.rec_path.setText(path)

    # 4 用户权限
    def tab_user(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("多角色用户权限管理")
        g_lay = QVBoxLayout(group)
        g_lay.setContentsMargins(12,16,12,12)
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(4)
        self.user_table.setHorizontalHeaderLabels(["ID","用户名","角色","权限描述"])
        self.user_table.verticalHeader().setDefaultSectionSize(26)
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)
        self.btn_add_user = QPushButton("新增用户")
        self.btn_del_user = QPushButton("删除选中账号")
        btn_box.addWidget(self.btn_add_user)
        btn_box.addWidget(self.btn_del_user)
        g_lay.addWidget(self.user_table)
        g_lay.addLayout(btn_box)
        lay.addWidget(group)
        lay.addStretch()
        return w

    # 5 OTA升级
    def tab_ota(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("OTA远程在线升级配置")
        form_lay = QFormLayout(group)
        form_lay.setContentsMargins(12,16,12,12)
        form_lay.setSpacing(12)
        self.ota_url = QLineEdit()
        self.ota_url.setPlaceholderText("升级包服务器地址")
        self.ota_crc = QCheckBox("升级包CRC完整性校验")
        self.ota_roll = QCheckBox("升级失败自动回滚旧版本")
        self.ota_safe = QCheckBox("升级过程不中断监控核心功能")
        form_lay.addRow("升级服务地址：", self.ota_url)
        form_lay.addRow("", self.ota_crc)
        form_lay.addRow("", self.ota_roll)
        form_lay.addRow("", self.ota_safe)
        lay.addWidget(group)
        lay.addStretch()
        return w

    # 6 NTP校时
    def tab_ntp(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)
        group = QGroupBox("网络NTP自动校时")
        form_lay = QFormLayout(group)
        form_lay.setContentsMargins(12,16,12,12)
        form_lay.setSpacing(12)
        self.ntp_server = QLineEdit()
        self.ntp_server.setPlaceholderText("推荐：ntp.aliyun.com")
        self.ntp_auto = QCheckBox("开机自动同步系统时间")
        self.label_sync = QLabel("上次同步：无同步记录")
        self.label_sync.setStyleSheet("color:#444;background:#fff2d8;padding:4px;border-radius:4px;")
        form_lay.addRow("NTP时间服务器：", self.ntp_server)
        form_lay.addRow("", self.ntp_auto)
        form_lay.addRow("同步状态：", self.label_sync)
        lay.addWidget(group)
        lay.addStretch()
        return w


    # 7 卡通主题页面
    def tab_theme(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(10)

        group = QGroupBox("UI卡通主题切换")
        g_lay = QVBoxLayout(group)
        g_lay.setContentsMargins(12,12,12,12)

        card_box = QHBoxLayout()
        card_box.setSpacing(12)
        self.theme_widget_map.clear()

        # 1 粉桃乐园（默认主界面粉色主题，读取 icons/pink.svg）
        card1 = ThemeCardFrame("粉桃乐园", self)
        c1_lay = QVBoxLayout(card1)
        lbl1 = QLabel('<img src="icons/pink" width="32" height="32" style="vertical-align:middle;margin-right:6px;">粉桃乐园')
        lbl1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl1.setStyleSheet("font-weight:bold;font-size:11pt;color:#c04060;")
        tip1 = QLabel("粉色奶油渐变，软件默认主题")
        tip1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip1.setStyleSheet("font-size:9pt;color:#333;")
        c1_lay.addWidget(lbl1)
        c1_lay.addWidget(tip1)
        self.theme_widget_map["粉桃乐园"] = card1

        # 2 太空宇航员
        card2 = ThemeCardFrame("太空宇航员", self)
        c2_lay = QVBoxLayout(card2)
        lbl2 = QLabel('<img src="icons/space.svg" width="32" height="32" style="vertical-align:middle;margin-right:6px;">太空宇航员')
        lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl2.setStyleSheet("font-weight:bold;font-size:11pt;color:#104090;")
        tip2 = QLabel("深蓝渐变+星空图标")
        tip2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip2.setStyleSheet("font-size:9pt;color:#333;")
        c2_lay.addWidget(lbl2)
        c2_lay.addWidget(tip2)
        self.theme_widget_map["太空宇航员"] = card2

        # 3 海底世界
        card3 = ThemeCardFrame("海底世界", self)
        c3_lay = QVBoxLayout(card3)
        lbl3 = QLabel('<img src="icons/ocean.svg" width="32" height="32" style="vertical-align:middle;margin-right:6px;">海底世界')
        lbl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl3.setStyleSheet("font-weight:bold;font-size:#006080;")
        tip3 = QLabel("浅蓝渐变+海洋图标")
        tip3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip3.setStyleSheet("font-size:9pt;color:#333;")
        c3_lay.addWidget(lbl3)
        c3_lay.addWidget(tip3)
        self.theme_widget_map["海底世界"] = card3

        card_box.addWidget(card1, stretch=1)
        card_box.addWidget(card2, stretch=1)
        card_box.addWidget(card3, stretch=1)
        tip_text = QLabel("⚠️ 切换主题后，重启软件界面生效")
        tip_text.setStyleSheet("color:#c03030;margin-top:8px;")
        g_lay.addLayout(card_box)
        g_lay.addWidget(tip_text)
        lay.addWidget(group)
        lay.addStretch()
        return w

    # 卡片点击回调
    def on_card_click(self, theme_name):
        self.cur_selected_theme = theme_name
        self.refresh_theme_card_style()

    # 更新卡片选中边框样式
    def refresh_theme_card_style(self):
        style_map = {
            "粉桃乐园": ("#ffb8cc", "#ff6699", "#ffe6ef"),
            "太空宇航员": ("#80b8ff", "#2070dd", "#e0edff"),
            "海底世界": ("#60c8e8", "#0098c8", "#d8f4fc")
        }
        for name, frame in self.theme_widget_map.items():
            base_border, sel_border, bg = style_map[name]
            if name == self.cur_selected_theme:
                frame.setStyleSheet(f"background:{bg};border:3px solid {sel_border};border-radius:10px;padding:12px;")
            else:
                frame.setStyleSheet(f"background:#f8f8f8;border:1px solid {base_border};border-radius:10px;padding:12px;")

    # 加载数据库配置
    def load_all_config(self):
        wake_data = self.db.get_wake_word()
        self.ed_w1.setText(wake_data["w1"])
        self.ed_w2.setText(wake_data["w2"])
        self.ed_w3.setText(wake_data["w3"])

        rec = self.db.get_record_cfg()
        self.rec_switch.setChecked(bool(rec[1]))
        self.rec_w.setValue(rec[2])
        self.rec_h.setValue(rec[3])
        self.rec_days.setValue(rec[4])
        self.rec_path.setText(rec[5])
        self.rec_pre.setChecked(bool(rec[6]))

        ota = self.db.get_ota()
        self.ota_url.setText(ota[1])
        self.ota_crc.setChecked(bool(ota[2]))
        self.ota_roll.setChecked(bool(ota[3]))
        self.ota_safe.setChecked(bool(ota[4]))

        ntp = self.db.get_ntp()
        self.ntp_server.setText(ntp[1])
        self.ntp_auto.setChecked(bool(ntp[2]))
        self.label_sync.setText(f"上次同步：{ntp[3]}  状态：{ntp[4]}")

        themes = self.db.get_all_theme()
        self.theme_map = {}
        for tid,name,cur in themes:
            self.theme_map[name] = tid
            if cur == 1:
                self.cur_selected_theme = name
        self.refresh_theme_card_style()

        alarm_data = self.db.get_all_alarm_strategy()
        self.alarm_table.setRowCount(len(alarm_data))
        for i,row in enumerate(alarm_data):
            typ,mode,en,lev,voi,pus,rec = row
            self.alarm_table.setItem(i,0,QTableWidgetItem(typ))
            self.alarm_table.setItem(i,1,QTableWidgetItem(mode))
            self.alarm_table.setItem(i,2,QTableWidgetItem("开启" if en else "关闭"))
            self.alarm_table.setItem(i,3,QTableWidgetItem(lev))
            self.alarm_table.setItem(i,4,QTableWidgetItem("发声" if voi else "静音"))
            self.alarm_table.setItem(i,5,QTableWidgetItem("推送" if pus else "关闭"))
            self.alarm_table.setItem(i,6,QTableWidgetItem("录制" if rec else "关闭"))

        users = self.db.get_all_users()
        self.user_table.setRowCount(len(users))
        for i,row in enumerate(users):
            uid,usr,role,perm = row
            self.user_table.setItem(i,0,QTableWidgetItem(str(uid)))
            self.user_table.setItem(i,1,QTableWidgetItem(usr))
            self.user_table.setItem(i,2,QTableWidgetItem(role))
            self.user_table.setItem(i,3,QTableWidgetItem(perm))

    # 保存配置
    def save_all(self):
        from PyQt6.QtWidgets import QMessageBox
        w1 = self.ed_w1.text().strip()
        w2 = self.ed_w2.text().strip()
        w3 = self.ed_w3.text().strip()
        if not w1:
            QMessageBox.warning(self,"提示","唤醒词1不能为空！")
            return
        self.db.update_wake_word(w1,w2,w3)
        self.db.update_record_cfg(
            1 if self.rec_switch.isChecked() else 0,
            self.rec_w.value(),
            self.rec_h.value(),
            self.rec_days.value(),
            self.rec_path.text(),
            1 if self.rec_pre.isChecked() else 0
        )
        self.db.update_ota(
            self.ota_url.text(),
            1 if self.ota_crc.isChecked() else 0,
            1 if self.ota_roll.isChecked() else 0,
            1 if self.ota_safe.isChecked() else 0
        )
        self.db.update_ntp(self.ntp_server.text(), 1 if self.ntp_auto.isChecked() else 0)
        if self.cur_selected_theme and self.cur_selected_theme in self.theme_map:
            sel_tid = self.theme_map[self.cur_selected_theme]
            self.db.set_current_theme(sel_tid)
            self.voice_player.voice_save_ok()
        QMessageBox.information(self,"保存成功","所有配置已写入数据库，主题切换重启软件生效！")

    def closeEvent(self, event):
        self.voice_player.voice_close_sys_settings()
        self.db.close()
        event.accept()