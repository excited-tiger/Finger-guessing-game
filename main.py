import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
                             QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QKeySequence, QShortcut
import random
import time
from gesture_recognizer import GestureRecognizer
from audio_processor import AudioProcessor


class GameState:
    WAITING = "waiting"
    READY = "ready"
    PLAYING = "playing"
    FINISHED = "finished"


class StyledLabel(QLabel):
    """自定义样式标签"""

    def __init__(self, text="", color="white", font_size=14, bold=False):
        super().__init__(text)
        font = QFont("PingFang SC", font_size)
        font.setBold(bold)
        self.setFont(font)
        self.setStyleSheet(f"color: {color}; padding: 5px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class StyledButton(QPushButton):
    """自定义样式按钮"""

    def __init__(self, text, color="#404040", hover_color="#505050"):
        super().__init__(text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                min-width: 120px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {hover_color};
                padding-top: 12px;
            }}
        """)


class InfoFrame(QFrame):
    """信息面板框架"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            InfoFrame {
                background-color: rgba(42, 42, 42, 0.9);
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)

        layout = QVBoxLayout(self)
        title_label = StyledLabel(title, color="#FFA500", font_size=16, bold=True)
        layout.addWidget(title_label)
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    gesture_detected = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, gesture_recognizer):
        super().__init__()
        self.running = True
        self.gesture_recognizer = gesture_recognizer
        self.cap = None
        self.fps_target = 30
        self.frame_time = 1.0 / self.fps_target
        self.last_frame_time = 0
        self.frame_count = 0
        self.last_fps_time = 0
        self.retry_count = 0
        self.max_retries = 3

    def initialize_camera(self):
        """初始化摄像头，失败时进行重试"""
        while self.retry_count < self.max_retries:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("无法打开摄像头")

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                print(f"摄像头初始化成功 - 分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
                return True

            except Exception as e:
                self.retry_count += 1
                print(f"摄像头初始化失败 (尝试 {self.retry_count}/{self.max_retries}): {str(e)}")
                if self.cap:
                    self.cap.release()
                time.sleep(1)

        self.error_occurred.emit("摄像头初始化失败，请检查设备连接")
        return False

    def run(self):
        if not self.initialize_camera():
            return

        self.last_fps_time = time.time()
        self.frame_count = 0

        while self.running:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            if elapsed < self.frame_time:
                time.sleep(max(0, self.frame_time - elapsed))
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("无法读取摄像头画面")

                gesture, processed_frame, landmarks_dict = self.gesture_recognizer.process_frame(frame.copy())
                if gesture is not None:
                    self.gesture_detected.emit(gesture, landmarks_dict if landmarks_dict else {})

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 1)
                self.frame_ready.emit(frame_rgb)

                self.frame_count += 1
                if current_time - self.last_fps_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_fps_time)
                    print(f"当前帧率: {fps:.1f} FPS")
                    self.frame_count = 0
                    self.last_fps_time = current_time

                self.last_frame_time = current_time

            except Exception as e:
                print(f"处理帧时出错: {e}")
                self.error_occurred.emit(f"摄像头错误: {str(e)}")
                if self.cap:
                    self.cap.release()
                if not self.initialize_camera():
                    break
                time.sleep(0.1)

    def stop(self):
        """停止摄像头线程"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class GameWindow(QMainWindow):
    """游戏主窗口"""

    def __init__(self):
        super().__init__()
        font = QFont("PingFang SC", 12)
        QApplication.setFont(font)
        self.init_game()
        self.init_ui()
        self.start_camera()

    def init_game(self):
        """初始化游戏状态"""
        self.state = GameState.WAITING
        self.score_player = 0
        self.score_computer = 0
        self.current_round = 0
        self.max_rounds = 5
        self.reaction_times = []
        self.player_number = None
        self.computer_number = None
        self.round_start_time = None
        # 新增变量用于区分手势和喊数，以及记录有效回合和操作记录
        self.player_gesture = None
        self.player_call = None
        self.computer_gesture = None
        self.computer_call = None
        self.valid_rounds = 0
        self.round_records = []
        self.score_draws = 0
        self.round_processed = False

        self.gesture_recognizer = GestureRecognizer()
        self.audio_processor = AudioProcessor(callback=self.on_voice_input)

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("十五二十猜拳游戏")
        self.showFullScreen()
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A1A;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # 使用水平布局：左侧为3D模型显示区，右侧为游戏UI和摄像头实时画面
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧：3D模型显示区域（中央区域较大）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.model_display = QLabel("3D手势模型显示")
        self.model_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_display.setStyleSheet("border: 2px solid #404040; border-radius: 10px; background-color: #FFFFFF;")
        self.model_display.setMinimumSize(800, 600)
        left_layout.addWidget(self.model_display)
        main_layout.addWidget(left_widget, stretch=3)

        # 右侧：分为上下两部分
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 右上部分：游戏UI（状态、控制、统计、操作记录）
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setSpacing(10)
        self.add_right_side_components(top_layout)
        right_layout.addWidget(top_widget, stretch=1)

        # 右下部分：摄像头实时画面
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.camera_display = QLabel("等待摄像头画面")
        self.camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_display.setStyleSheet("border: 2px solid #404040; border-radius: 10px; background-color: #000000;")
        self.camera_display.setMinimumSize(400, 300)
        bottom_layout.addWidget(self.camera_display)
        right_layout.addWidget(bottom_widget, stretch=1)

        main_layout.addWidget(right_widget, stretch=1)

    def add_right_side_components(self, right_layout):
        """添加右侧组件（右上角）"""
        # 状态显示
        status_frame = InfoFrame("游戏状态")
        self.status_label = StyledLabel("等待开始...", color="#FFD700", font_size=18, bold=True)
        status_frame.add_widget(self.status_label)
        right_layout.addWidget(status_frame)

        # 控制按钮
        control_frame = InfoFrame("游戏控制")
        button_layout = QHBoxLayout()
        self.start_button = StyledButton("开始游戏", color="#28A745", hover_color="#218838")
        self.exit_button = StyledButton("退出游戏", color="#DC3545", hover_color="#C82333")
        self.rule_button = StyledButton("游戏规则", color="#007BFF", hover_color="#0056b3")
        self.start_button.clicked.connect(self.start_game)
        self.exit_button.clicked.connect(self.exit_game)
        self.rule_button.clicked.connect(self.show_rules)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.rule_button)
        control_frame.content_layout.addLayout(button_layout)
        right_layout.addWidget(control_frame)

        # 实时统计
        score_frame = InfoFrame("实时统计")
        score_layout = QHBoxLayout()
        self.player_score_label = StyledLabel("玩家: 0", color="#00FF00", font_size=16)
        self.computer_score_label = StyledLabel("电脑: 0", color="#FF4444", font_size=16)
        self.draw_score_label = StyledLabel("平局: 0", color="#FFFF00", font_size=16)
        score_layout.addWidget(self.player_score_label)
        score_layout.addWidget(self.computer_score_label)
        score_layout.addWidget(self.draw_score_label)
        score_frame.content_layout.addLayout(score_layout)
        right_layout.addWidget(score_frame)

        # 操作记录
        record_frame = InfoFrame("操作记录")
        self.record_label = StyledLabel("暂无记录", color="white", font_size=14)
        record_frame.add_widget(self.record_label)
        right_layout.addWidget(record_frame)

        right_layout.addStretch()

    def start_camera(self):
        """启动摄像头"""
        self.camera_thread = CameraThread(self.gesture_recognizer)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.gesture_detected.connect(self.on_gesture_detected)
        self.camera_thread.error_occurred.connect(self.on_camera_error)
        self.camera_thread.start()

    def on_camera_error(self, error_msg):
        """处理摄像头错误"""
        self.status_label.setText(error_msg)
        self.status_label.setStyleSheet("color: #FF4444; font-size: 18px; font-weight: bold;")

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """更新摄像头画面"""
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            label_size = self.camera_display.size()
            scaled_size = qt_image.size()
            scaled_size.scale(label_size, Qt.AspectRatioMode.KeepAspectRatio)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation)
            self.camera_display.setPixmap(scaled_pixmap)
            x = (label_size.width() - scaled_size.width()) // 2
            y = (label_size.height() - scaled_size.height()) // 2
            self.camera_display.setContentsMargins(x, y, x, y)
        except Exception as e:
            print(f"更新帧时出错: {e}")

    def start_game(self):
        """开始游戏"""
        self.state = GameState.PLAYING
        self.score_player = 0
        self.score_computer = 0
        self.score_draws = 0
        self.valid_rounds = 0
        self.reaction_times.clear()
        self.round_records.clear()
        self.status_label.setText("游戏开始！")
        self.status_label.setStyleSheet("color: #00FF00; font-size: 18px; font-weight: bold;")

        self.gesture_recognizer.start()
        self.audio_processor.start()

        # 显示3秒准备动画，然后开始回合
        self.status_label.setText("准备中...")
        QTimer.singleShot(3000, self.start_round)

    def start_round(self):
        """开始新的回合，设置1秒前静止状态，并启动15秒超时计时器"""
        self.round_processed = False
        self.player_gesture = None
        self.player_call = None
        self.computer_gesture = None
        self.computer_call = None
        self.round_start_time = time.time()
        self.status_label.setText("等待输入...")
        self.status_label.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold;")
        self.update_ui()

        # 启动15秒超时计时器，若玩家未输入则自动生成
        QTimer.singleShot(15000, self.auto_process_round)

    def auto_process_round(self):
        """15秒内未处理输入，自动生成有效输入"""
        if self.state != GameState.PLAYING or self.round_processed:
            return
        if self.player_gesture is None:
            self.player_gesture = random.choice(["0", "5"])
        if self.player_call is None:
            self.player_call = random.choice([15, 20])
        self.process_round()

    def process_round(self):
        """处理本回合逻辑"""
        if self.player_gesture is None or self.player_call is None:
            return

        # 确保出拳前有1秒静止
        elapsed = time.time() - self.round_start_time
        if elapsed < 1:
            remaining = int((1 - elapsed) * 1000)
            QTimer.singleShot(remaining, self.process_round)
            return

        self.round_processed = True

        # 电脑随机选择手势与喊数，0和5各50%
        self.computer_gesture = random.choice(["0", "5"])
        self.computer_call = random.choice([15, 20])

        def convert_gesture(g):
            return int(g)

        player_value = convert_gesture(self.player_gesture)
        computer_value = convert_gesture(self.computer_gesture)
        total = player_value + computer_value

        if self.player_call == total and self.player_call != self.computer_call:
            outcome = "胜利！"
            self.score_player += 1
        elif self.computer_call == total and self.player_call != self.computer_call:
            outcome = "失败..."
            self.score_computer += 1
        else:
            outcome = "平局"
            self.score_draws += 1

        if self.round_start_time:
            reaction_time = time.time() - self.round_start_time
            self.reaction_times.append(reaction_time)

        if outcome != "平局":
            self.valid_rounds += 1

        record = f"玩家: 手势[{self.player_gesture}], 喊数[{self.player_call}] | 电脑: 手势[{self.computer_gesture}], 喊数[{self.computer_call}] | 总和: {total} | 结果: {outcome}"
        self.round_records.append(record)
        self.round_records = self.round_records[-3:]

        if outcome == "胜利！":
            self.status_label.setText(outcome)
            self.status_label.setStyleSheet("color: #00FF00; font-size: 18px; font-weight: bold;")
        elif outcome == "失败...":
            self.status_label.setText(outcome)
            self.status_label.setStyleSheet("color: #FF4444; font-size: 18px; font-weight: bold;")
        else:
            self.status_label.setText(outcome)
            self.status_label.setStyleSheet("color: #AAAAAA; font-size: 18px; font-weight: bold;")

        self.update_ui(total)

        if self.valid_rounds >= self.max_rounds:
            QTimer.singleShot(2000, self.end_game)
        else:
            QTimer.singleShot(2000, self.start_round)

    def update_ui(self, total=None):
        """更新界面显示"""
        self.player_score_label.setText(f"玩家: {self.score_player}")
        self.computer_score_label.setText(f"电脑: {self.score_computer}")
        self.round_label.setText(f"回合: {self.valid_rounds}/{self.max_rounds}  平局: {self.score_draws}")

        if self.player_gesture is not None and self.player_call is not None:
            self.player_number_label.setText(f"玩家手势: {self.player_gesture}, 喊数: {self.player_call}")
        else:
            self.player_number_label.setText("玩家: 等待输入")

        if self.computer_gesture is not None and self.computer_call is not None:
            self.computer_number_label.setText(f"电脑手势: {self.computer_gesture}, 喊数: {self.computer_call}")
        else:
            self.computer_number_label.setText("电脑: 等待输入")

        if total is not None:
            self.total_label.setText(f"总和: {total}")
        else:
            self.total_label.setText("总和: --")

        if hasattr(self, 'record_label'):
            if self.round_records:
                self.record_label.setText("\n".join(self.round_records))
            else:
                self.record_label.setText("暂无记录")

    def end_game(self):
        """结束游戏并显示战报"""
        self.state = GameState.FINISHED
        self.gesture_recognizer.stop()
        self.audio_processor.stop()
        avg_time = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0
        report = f"游戏结束！\n玩家胜局: {self.score_player}\n电脑胜局: {self.score_computer}\n平局: {self.score_draws}\n平均反应时间: {avg_time:.2f}秒"
        from PyQt6.QtWidgets import QMessageBox
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("战报")
        msg_box.setText(report)
        continue_button = msg_box.addButton("继续", QMessageBox.ButtonRole.AcceptRole)
        exit_button = msg_box.addButton("退出", QMessageBox.ButtonRole.RejectRole)
        msg_box.exec()
        if msg_box.clickedButton() == continue_button:
            self.start_game()
        else:
            self.close()

    def exit_game(self):
        """退出游戏，返回主界面"""
        self.close()

    def show_rules(self):
        """显示游戏规则"""
        from PyQt6.QtWidgets import QMessageBox
        rules = ("游戏规则：\n"
                 "1. 游戏模式：人机对战，5局有效比赛（平局不计入）。\n"
                 "2. 玩家通过语音或手势输入。\n"
                 "   - 手势：握拳代表0，手掌代表5。\n"
                 "   - 语音：喊出数字（15或20）作为猜测总和。\n"
                 "3. 每局限时15秒，出拳前保持1秒静止。\n"
                 "4. 胜利条件：玩家喊数等于双方手势总和且与电脑喊数不同则获胜；电脑相反则获胜；否则平局。\n"
                 "5. 结束后显示战报，提供继续或退出选项。")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("游戏规则")
        msg_box.setText(rules)
        msg_box.exec()

    def on_gesture_detected(self, gesture, landmarks_dict=None):
        """处理手势识别结果"""
        if self.state == GameState.PLAYING:
            mapping = {"Fist": "0", "Palm": "5"}
            new_gesture = mapping.get(gesture, None)
            if new_gesture is None:
                return    # 忽略不在要求范围内的手势
            self.gesture_label.setText(f"手势: {new_gesture}")
            self.player_gesture = new_gesture

            # 更新3D模型显示
            if landmarks_dict:
                self.update_3d_model(new_gesture, landmarks_dict)

            if self.player_call is not None:
                self.process_round()

    def on_voice_input(self, text):
        """处理语音输入"""
        if self.state == GameState.WAITING and ("开始" in text or "15 15" in text):
            self.start_game()
            return

        if "退出游戏" in text:
            self.exit_game()
            return

        self.voice_label.setText(f"语音: {text}")
        number = self.audio_processor.extract_number(text)
        if number is not None:
            self.player_call = number
            if self.player_gesture is not None:
                self.process_round()

    def closeEvent(self, event):
        """关闭窗口事件"""
        try:
            if hasattr(self, 'camera_thread'):
                self.camera_thread.stop()
            if hasattr(self, 'gesture_recognizer'):
                self.gesture_recognizer.stop()
            if hasattr(self, 'audio_processor'):
                self.audio_processor.stop()
        except Exception as e:
            print(f"关闭时出错: {e}")
        event.accept()

    def update_3d_model(self, gesture, landmarks_dict):
        """
        根据实时识别的手势和坐标信息更新3D模型显示
        Args:
            gesture: 识别出的手势
            landmarks_dict: 手部关键点的3D坐标信息
        """
        # 构建显示文本
        display_text = [f"3D Model - Current Gesture: {gesture}"]

        # 添加手部关键点坐标信息
        for hand_side, landmarks in landmarks_dict.items():
            display_text.append(f"\n{hand_side} Hand Landmarks:")
            for i, landmark in enumerate(landmarks):
                display_text.append(f"Point {i}: x={landmark['x']:.3f}, y={landmark['y']:.3f}, z={landmark['z']:.3f}")

        # 更新显示
        self.model_display.setText("\n".join(display_text))

        # TODO: 在这里可以添加实际的3D模型渲染代码
        # 可以使用 PyOpenGL 或其他3D渲染库来实现真实的3D手势模型


if __name__ == "__main__":
    app = QApplication(sys.argv)

    def toggle_fullscreen(window):
        if window.isFullScreen():
            window.showNormal()
        else:
            window.showFullScreen()

    window = GameWindow()
    shortcut = QShortcut(QKeySequence("Esc"), window)
    shortcut.activated.connect(lambda: toggle_fullscreen(window))

    window.show()
    sys.exit(app.exec())
