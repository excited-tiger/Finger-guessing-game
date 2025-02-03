import time
import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe 手势识别器
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 手势置信度阈值
GESTURE_CONFIDENCE_THRESHOLD = {
    "Closed_Fist": 0.1,    # 拳头的置信度阈值
    "Open_Palm": 0.1,    # 手掌的置信度阈值
    "Pointing_Up": 0.5,
    "Thumb_Down": 0.5,
    "Thumb_Up": 0.5,
    "Victory": 0.5,
    "ILoveYou": 0.5
}

# 手势名称映射（英文显示）
GESTURE_NAME_MAPPING = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Palm",
    "Pointing_Up": "Point Up",
    "Thumb_Down": "Thumb Down",
    "Thumb_Up": "Thumb Up",
    "Victory": "Victory",
    "ILoveYou": "Love",
    "None": "Unknown"
}

# 用于存储结果的全局变量
recognition_result_list = []


def save_result(result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result_list
    recognition_result_list.append(result)


def filter_gesture(gesture, score):
    """过滤低置信度的手势识别结果"""
    if gesture in GESTURE_CONFIDENCE_THRESHOLD:
        if score < GESTURE_CONFIDENCE_THRESHOLD[gesture]:
            return "None"
    return gesture


# 创建手势识别器
base_options = BaseOptions(model_asset_path='gesture_recognizer.task')
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,    # 设置最大检测手数为2
    min_hand_detection_confidence=0.3,    # 降低检测阈值，提高握拳检测率
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    result_callback=save_result)
recognizer = GestureRecognizer.create_from_options(options)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 用于控制处理帧率的变量
frame_timestamp = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 转换图像格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # 进行手势识别
    frame_timestamp += 1
    recognizer.recognize_async(mp_image, frame_timestamp)

    # 绘制结果
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 先翻转图像
    image = cv2.flip(image, 1)

    if recognition_result_list:
        current_result = recognition_result_list[-1]

        # 显示手势
        if current_result.gestures and len(current_result.gestures) > 0:
            for idx, (gesture, handedness) in enumerate(zip(current_result.gestures, current_result.handedness)):
                gesture_category = gesture[0].category_name
                gesture_score = gesture[0].score

                # 过滤低置信度的手势
                filtered_gesture = filter_gesture(gesture_category, gesture_score)

                # 转换为英文显示
                display_gesture = GESTURE_NAME_MAPPING.get(filtered_gesture, filtered_gesture)

                hand_side = "Right" if handedness[0].category_name == "Right" else "Left"
                y_position = 30 + idx * 30
                text = f"{hand_side} Hand - {display_gesture}: {gesture_score:.2f}"
                cv2.putText(image, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示手部关键点
        if current_result.hand_landmarks:
            image_height, image_width, _ = image.shape
            for hand_landmarks, handedness in zip(current_result.hand_landmarks, current_result.handedness):
                # 将关键点坐标转换为像素坐标（注意要翻转x坐标）
                landmark_points = []
                for landmark in hand_landmarks:
                    x = int((1 - landmark.x) * image_width)    # 翻转x坐标
                    y = int(landmark.y * image_height)
                    landmark_points.append((x, y))

                # 根据左右手设置不同颜色
                color = (0, 255, 0) if handedness[0].category_name == "Right" else (255, 0, 0)

                # 绘制关键点和连接线
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv2.line(image, landmark_points[start_idx], landmark_points[end_idx], color, 2)

                for point in landmark_points:
                    cv2.circle(image, point, 5, (0, 0, 255), -1)

    # 显示图像
    cv2.imshow('MediaPipe Gesture Recognition', image)

    # 清空结果列表，只保留最新的结果
    recognition_result_list.clear()

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
recognizer.close()
cv2.destroyAllWindows()
