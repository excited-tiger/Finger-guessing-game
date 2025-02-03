import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict
import time
from mediapipe.framework.formats import landmark_pb2

# 初始化 MediaPipe 手势识别器
BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 手势置信度阈值
GESTURE_CONFIDENCE_THRESHOLD = 0.5

# 手势名称映射
GESTURE_NAME_MAPPING = {
    "Closed_Fist": "0",    # 拳头映射为 0
    "Open_Palm": "5",    # 手掌映射为 5
}


class GestureRecognizer:

    def __init__(self):
        # 创建手势识别器
        base_options = BaseOptions(model_asset_path='gesture_recognizer.task')
        options = GestureRecognizerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,    # 使用图像模式
            num_hands=2,    # 最多检测两只手
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.2)
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray, Optional[Dict]]:
        """处理单帧图像"""
        # 创建一个更大的画布（保持原始图像比例）
        height, width = frame.shape[:2]
        scale_factor = 2    # 放大2倍
        canvas_height = height * scale_factor
        canvas_width = width * scale_factor

        # 创建黑色画布
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 将原始图像放大并放置在画布中央
        frame_hd = cv2.resize(frame, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LANCZOS4)
        canvas[0:frame_hd.shape[0], 0:frame_hd.shape[1]] = frame_hd

        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame_hd, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 处理图像
        recognition_result = self.recognizer.recognize(mp_image)

        # 在图像上绘制结果
        frame_out = canvas.copy()
        landmarks_dict = None
        detected_numbers = []    # 存储检测到的数字

        if recognition_result.hand_landmarks:
            landmarks_dict = {}

            for idx, (hand_landmarks,
                      handedness) in enumerate(zip(recognition_result.hand_landmarks, recognition_result.handedness)):
                # 获取手的类型（左手或右手）
                hand_side = "Right" if handedness[0].category_name == "Right" else "Left"

                # 记录关键点坐标
                landmarks = []
                for landmark in hand_landmarks:
                    landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
                landmarks_dict[hand_side] = landmarks

                # 获取手势类型
                if recognition_result.gestures and len(recognition_result.gestures) > idx:
                    gesture_category = recognition_result.gestures[idx][0].category_name
                    gesture_score = recognition_result.gestures[idx][0].score

                    # 过滤低置信度的手势
                    if gesture_score >= GESTURE_CONFIDENCE_THRESHOLD:
                        gesture = GESTURE_NAME_MAPPING.get(gesture_category)
                        if gesture:
                            detected_numbers.append(int(gesture))

                # 绘制手部关键点
                self._draw_hand_landmarks(frame_out, hand_landmarks, hand_side)

            # 计算并显示手势数字之和
            if detected_numbers:
                total = sum(detected_numbers)
                # 准备文字
                text = f"Total: {total}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1 * scale_factor
                thickness = 5

                # 获取文字大小
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # 创建文字图层（带背景色）
                text_layer = np.zeros((text_height + baseline + 20, text_width + 20, 3), dtype=np.uint8)
                # 添加半透明黑色背景
                cv2.rectangle(text_layer, (0, 0), (text_width + 20, text_height + baseline + 20), (0, 0, 0), -1)
                # 添加文字
                cv2.putText(text_layer, text, (10, text_height + 10), font, font_scale, (0, 255, 0), thickness)

                # 水平翻转文字图层
                text_layer = cv2.flip(text_layer, 1)

                # 将文字叠加到图像上（左上角）
                x_pos = 5 * scale_factor
                y_pos = 5 * scale_factor
                # 使背景半透明
                alpha = 0.7
                roi = frame_out[y_pos:y_pos + text_height + baseline + 20, x_pos:x_pos + text_width + 20]
                cv2.addWeighted(text_layer, alpha, roi, 1 - alpha, 0, roi)
                frame_out[y_pos:y_pos + text_height + baseline + 20, x_pos:x_pos + text_width + 20] = roi

        else:
            # 如果没有检测到手，显示翻转的提示信息
            text = "No hand detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1 * scale_factor
            thickness = 5

            # 获取文字大小
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # 创建文字图层（带背景色）
            text_layer = np.zeros((text_height + baseline + 20, text_width + 20, 3), dtype=np.uint8)
            # 添加半透明黑色背景
            cv2.rectangle(text_layer, (0, 0), (text_width + 20, text_height + baseline + 20), (0, 0, 0), -1)
            # 添加文字
            cv2.putText(text_layer, text, (10, text_height + 10), font, font_scale, (0, 0, 255), thickness)

            # 水平翻转文字图层
            text_layer = cv2.flip(text_layer, 1)

            # 将文字叠加到图像上（左上角）
            x_pos = 5 * scale_factor
            y_pos = 5 * scale_factor
            # 使背景半透明
            alpha = 0.8
            roi = frame_out[y_pos:y_pos + text_height + baseline + 20, x_pos:x_pos + text_width + 20]
            cv2.addWeighted(text_layer, alpha, roi, 1 - alpha, 0, roi)
            frame_out[y_pos:y_pos + text_height + baseline + 20, x_pos:x_pos + text_width + 20] = roi

        # 将输出图像缩放回原始大小
        frame_out = cv2.resize(frame_out, (width, height), interpolation=cv2.INTER_AREA)

        return str(sum(detected_numbers)) if detected_numbers else None, frame_out, landmarks_dict

    def _draw_hand_landmarks(self, image, landmarks, hand_side):
        """绘制手部关键点和连接线"""
        # 转换为 mediapipe 的 NormalizedLandmarkList 格式
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for landmark in landmarks:
            landmark_list.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z)

        # 设置颜色：右手绿色，左手红色
        color = (0, 255, 0) if hand_side == "Right" else (0, 0, 255)

        # 自定义绘制参数
        scale_factor = 2    # 与图像放大倍数一致
        landmark_spec = self.mp_drawing.DrawingSpec(color=color,
                                                    thickness=4 * scale_factor,
                                                    circle_radius=4 * scale_factor)
        connection_spec = self.mp_drawing.DrawingSpec(color=color,
                                                      thickness=3 * scale_factor,
                                                      circle_radius=3 * scale_factor)

        # 绘制关键点和连接线
        self.mp_drawing.draw_landmarks(image, landmark_list, self.mp_hands.HAND_CONNECTIONS, landmark_spec,
                                       connection_spec)

    def start(self):
        """启动手势识别器"""
        pass

    def stop(self):
        """停止手势识别器"""
        self.recognizer.close()
