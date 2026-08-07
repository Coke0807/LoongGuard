import sqlite3
import json
from typing import List, Dict, Any

DB_PATH = "./system_settings.db"

class SettingsDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_all_tables()
        self.init_default_data()

    def create_all_tables(self):
        # 1.唤醒词配置 支持1-3个
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS wake_word (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word1 TEXT NOT NULL,
            word2 TEXT,
            word3 TEXT
        )
        ''')
        # 2.告警策略（告警类型+三种模式独立配置）
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS alarm_strategy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alarm_type TEXT NOT NULL,
            mode TEXT NOT NULL,
            enable INTEGER DEFAULT 0,
            level TEXT DEFAULT "普通",
            voice INTEGER DEFAULT 0,
            push INTEGER DEFAULT 0,
            record INTEGER DEFAULT 0,
            UNIQUE(alarm_type, mode)
        )
        ''')
        # 3.视频录制全局配置
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS record_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_switch INTEGER DEFAULT 1,
            width INTEGER DEFAULT 1280,
            height INTEGER DEFAULT 720,
            save_days INTEGER DEFAULT 30,
            save_path TEXT DEFAULT "./video_record",
            pre_record INTEGER DEFAULT 0
        )
        ''')
        # 4.用户权限角色
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_role (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            permission_json TEXT
        )
        ''')
        # 5.OTA升级配置
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ota_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_url TEXT,
            check_crc INTEGER DEFAULT 1,
            rollback INTEGER DEFAULT 1,
            core_safe INTEGER DEFAULT 1
        )
        ''')
        # 6.NTP校时配置
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ntp_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ntp_server TEXT DEFAULT "ntp.aliyun.com",
            auto_sync INTEGER DEFAULT 1,
            last_sync_time TEXT,
            sync_status TEXT
        )
        ''')
        # 7.UI卡通主题
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ui_theme (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme_name TEXT UNIQUE NOT NULL,
            bg_style TEXT,
            icon_set TEXT,
            current_use INTEGER DEFAULT 0
        )
        ''')
        self.conn.commit()

    def init_default_data(self):
        # 默认唤醒词
        self.cursor.execute("SELECT COUNT(*) FROM wake_word")
        if self.cursor.fetchone()[0] == 0:
            self.cursor.execute("INSERT INTO wake_word(word1,word2,word3) VALUES(?,?,?)", ("小卫士", "", ""))
        # 默认三种模式+基础告警类型
        modes = ["公共巡查模式","课堂模式","午睡模式"]
        alarm_types = ["危险品预警","奔跑预警","环境异常","摄像头离线","网络断开"]
        for m in modes:
            for t in alarm_types:
                self.cursor.execute("INSERT OR IGNORE INTO alarm_strategy(alarm_type,mode) VALUES(?,?)", (t,m))
        # 默认录制
        self.cursor.execute("SELECT COUNT(*) FROM record_config")
        if self.cursor.fetchone()[0]==0:
            self.cursor.execute('INSERT INTO record_config VALUES(null,1,1280,720,30,"./video_record",0)')
        # 默认管理员账号
        self.cursor.execute("SELECT COUNT(*) FROM user_role")
        if self.cursor.fetchone()[0]==0:
            perm = json.dumps({"all":1})
            self.cursor.execute("INSERT INTO user_role(username,password,role,permission_json) VALUES(?,?,?,?)",
                                ("admin","123456","超级管理员",perm))
        # 默认OTA
        self.cursor.execute("SELECT COUNT(*) FROM ota_config")
        if self.cursor.fetchone()[0]==0:
            self.cursor.execute('INSERT INTO ota_config VALUES(null,"http://upgrade.xxx.com",1,1,1)')
        # 默认NTP
        self.cursor.execute("SELECT COUNT(*) FROM ntp_config")
        if self.cursor.fetchone()[0]==0:
            self.cursor.execute('INSERT INTO ntp_config VALUES(null,"ntp.aliyun.com",1,"","未同步")')
        # 默认主题
        self.cursor.execute("SELECT COUNT(*) FROM ui_theme")
        themes = [
            ("森林小动物","浅绿渐变","animal",1),
            ("太空宇航员","深蓝渐变","space",0),
            ("海底世界","浅蓝渐变","sea",0)
        ]
        if self.cursor.fetchone()[0]==0:
            for name,bg,icon,cur in themes:
                self.cursor.execute("INSERT INTO ui_theme(theme_name,bg_style,icon_set,current_use) VALUES(?,?,?,?)",
                                    (name,bg,icon,cur))
        self.conn.commit()

    # ========== 唤醒词 CRUD ==========
    def get_wake_word(self) -> dict:
        self.cursor.execute("SELECT word1,word2,word3 FROM wake_word LIMIT 1")
        row = self.cursor.fetchone()
        return {"w1":row[0],"w2":row[1],"w3":row[2]}
    def update_wake_word(self,w1,w2,w3):
        self.cursor.execute("UPDATE wake_word SET word1=?,word2=?,word3=? WHERE id=1",(w1,w2,w3))
        self.conn.commit()

    # ========== 告警策略 ==========
    def get_all_alarm_strategy(self):
        self.cursor.execute("SELECT alarm_type,mode,enable,level,voice,push,record FROM alarm_strategy")
        return self.cursor.fetchall()
    def update_alarm_strategy(self,alarm_type,mode,enable,level,voice,push,record):
        self.cursor.execute('''
        UPDATE alarm_strategy SET enable=?,level=?,voice=?,push=?,record=?
        WHERE alarm_type=? AND mode=?
        ''',(enable,level,voice,push,record,alarm_type,mode))
        self.conn.commit()

    # ========== 录制配置 ==========
    def get_record_cfg(self):
        self.cursor.execute("SELECT * FROM record_config LIMIT 1")
        return self.cursor.fetchone()
    def update_record_cfg(self,switch,w,h,days,save_path,pre):
        self.cursor.execute('''
        UPDATE record_config SET global_switch=?,width=?,height=?,save_days=?,save_path=?,pre_record=?
        WHERE id=1
        ''',(switch,w,h,days,save_path,pre))
        self.conn.commit()

    # ========== 用户角色 ==========
    def get_all_users(self):
        self.cursor.execute("SELECT id,username,role,permission_json FROM user_role")
        return self.cursor.fetchall()
    def add_user(self,user,pwd,role,perm_json):
        try:
            self.cursor.execute("INSERT INTO user_role(username,password,role,permission_json) VALUES(?,?,?)",
                                (user,pwd,role,perm_json))
            self.conn.commit()
            return True
        except:
            return False
    def del_user(self,uid):
        self.cursor.execute("DELETE FROM user_role WHERE id=?",(uid,))
        self.conn.commit()

    # ========== OTA ==========
    def get_ota(self):
        self.cursor.execute("SELECT * FROM ota_config LIMIT 1")
        return self.cursor.fetchone()
    def update_ota(self,url,crc,rollback,safe):
        self.cursor.execute("UPDATE ota_config SET server_url=?,check_crc=?,rollback=?,core_safe=? WHERE id=1",
                            (url,crc,rollback,safe))
        self.conn.commit()

    # ========== NTP校时 ==========
    def get_ntp(self):
        self.cursor.execute("SELECT * FROM ntp_config LIMIT 1")
        return self.cursor.fetchone()
    def update_ntp(self,server,auto):
        self.cursor.execute("UPDATE ntp_config SET ntp_server=?,auto_sync=? WHERE id=1",(server,auto))
        self.conn.commit()
    def set_ntp_sync_status(self,time_str,status):
        self.cursor.execute("UPDATE ntp_config SET last_sync_time=?,sync_status=? WHERE id=1",(time_str,status))
        self.conn.commit()

    # ========== UI主题 ==========
    def get_all_theme(self):
        self.cursor.execute("SELECT id,theme_name,current_use FROM ui_theme")
        return self.cursor.fetchall()
    def set_current_theme(self,tid):
        self.cursor.execute("UPDATE ui_theme SET current_use=0")
        self.cursor.execute("UPDATE ui_theme SET current_use=1 WHERE id=?",(tid,))
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
