# 十五二十猜拳游戏 - 技术需求与设计文档

## 1. 系统架构设计

### 1.1 整体架构
- **设计模式**：MVC架构
  - Model：游戏逻辑和数据处理
  - View：Qt6 GUI界面
  - Controller：事件处理和业务流程控制

### 1.2 核心模块
1. **GUI模块**
   - 基于Qt6框架
   - 实现主界面和游戏界面
   - 负责手势渲染和状态显示

2. **手势识别模块**
   - 基于MediaPipe框架
   - 实时手部关键点检测
   - 手势分类（握拳/张开）

3. **语音识别模块**
   - 基于FunASR框架
   - 实时语音指令识别
   - 语音文本转换

4. **游戏逻辑模块**
   - 状态管理
   - 胜负判定
   - 数据统计

## 2. 详细技术规格

### 2.1 开发环境
- **编程语言**：Python 3.9+
- **主要依赖**：
  - Qt6：GUI框架
  - MediaPipe：手势识别
  - FunASR：语音识别
  - OpenCV：图像处理
  - NumPy：数值计算

### 2.2 模块技术规格

#### 2.2.1 GUI模块
- **框架**：PyQt6
- **窗口分辨率**：1920x1080
- **刷新率**：60fps
- **主要组件**：
  - 主画面（QMainWindow）
  - 视频显示区（QLabel）
  - 状态显示区（QLabel）
  - 控制按钮（QPushButton）
  - 数据面板（QWidget）

#### 2.2.2 手势识别模块
- **框架选型**：MediaPipe Hands
- **核心功能**：
  1. 手部检测与追踪
     - 双手21点骨骼实时追踪
     - 3D坐标系转换与映射
     - 手部区域ROI提取
  
  2. 手势状态分析
     - 手指关节角度计算
     - 手掌开合度评估
     - 手势状态分类（0/5）

  3. 手势判定算法
     ```python
     def calculate_hand_state(landmarks):
         # 计算手指关节角度
         finger_angles = []
         for finger in range(5):
             angle = calculate_finger_angle(landmarks, finger)
             finger_angles.append(angle)
         
         # 判定手势状态
         if avg(finger_angles) < THRESHOLD_CLOSED:
             return 0  # 握拳
         else:
             return 5  # 张开
     ```

  4. 性能优化
     - 帧间追踪优化
     - ROI区域动态调整
     - 多线程并行处理

- **性能指标**：
  - 识别延迟：<100ms
  - 最小手部尺寸：50x50像素
  - 置信度阈值：0.8
  - 识别准确率：>95%

- **异常处理**：
  - 手部遮挡处理
  - 光线变化适应
  - 手势过渡状态处理

#### 2.2.3 语音识别模块
- **框架选型**：FunASR
- **核心功能**：
  1. 音频采集与预处理
     - 实时音频流获取
     - 音频降噪处理
     - 音频分帧与特征提取

  2. 语音识别引擎
     - 声学模型处理
     - 语言模型匹配
     - 指令词识别

  3. 语音识别结果处理
     ```python
     class VoiceResultProcessor:
         def __init__(self):
             # 历史文本缓存
             self.history_text = ""
             # 文本清理时间窗口（毫秒）
             self.clear_window = 1500
             # 上次清理时间
             self.last_clear_time = 0
             # 数字映射字典
             self.number_mapping = {
                 "零": 0, "〇": 0, "0": 0,
                 "五": 5, "5": 5,
                 "十五": 15, "一五": 15, "15": 15,
                 "二十": 20, "20": 20
             }
             # 指令关键词
             self.command_keywords = {
                 "开始游戏": "START_GAME",
                 "结束游戏": "END_GAME",
                 "退出游戏": "EXIT_GAME",
                 "十五二十": "GAME_READY"
             }
         
         def process_stream_result(self, new_text: str) -> dict:
             """处理流式识别结果"""
             current_time = time.time() * 1000
             
             # 检查是否需要清理历史
             if current_time - self.last_clear_time > self.clear_window:
                 self.history_text = ""
             
             # 更新历史文本
             self.history_text += new_text
             
             # 处理结果
             result = {
                 'command': None,  # 识别到的指令
                 'number': None,   # 识别到的数字
                 'should_clear': False  # 是否需要清理历史
             }
             
             # 检查指令关键词
             for keyword, command in self.command_keywords.items():
                 if keyword in self.history_text:
                     result['command'] = command
                     result['should_clear'] = True
                     break
             
             # 检查数字
             if not result['command']:
                 for text, number in self.number_mapping.items():
                     if text in self.history_text:
                         result['number'] = number
                         result['should_clear'] = True
                         break
             
             # 如果需要清理，更新时间戳
             if result['should_clear']:
                 self.last_clear_time = current_time
                 self.history_text = ""
             
             return result
         
         def handle_noise_and_errors(self, text: str) -> str:
             """处理噪音和错误"""
             # 移除标点符号
             text = re.sub(r'[^\w\s]', '', text)
             # 移除多余空格
             text = ' '.join(text.split())
             return text
         
         def validate_number(self, number: int) -> bool:
             """验证数字是否有效"""
             return number in [0, 5, 15, 20]
     ```

  4. 语音处理状态机
     ```python
     class VoiceProcessState:
         IDLE = "IDLE"           # 空闲状态
         LISTENING = "LISTENING" # 正在监听
         PROCESSING = "PROCESSING" # 正在处理
         COMMAND_DETECTED = "COMMAND_DETECTED" # 检测到指令
         NUMBER_DETECTED = "NUMBER_DETECTED"   # 检测到数字
         
         def __init__(self):
             self.current_state = self.IDLE
             self.state_timestamp = time.time()
             self.detection_window = 1.5  # 1.5秒检测窗口
         
         def update_state(self, result: dict) -> str:
             """更新状态机状态"""
             current_time = time.time()
             
             # 状态超时检查
             if current_time - self.state_timestamp > self.detection_window:
                 self.current_state = self.IDLE
             
             # 根据识别结果更新状态
             if result['command']:
                 self.current_state = self.COMMAND_DETECTED
             elif result['number'] is not None:
                 self.current_state = self.NUMBER_DETECTED
             
             self.state_timestamp = current_time
             return self.current_state
     ```

  5. 鲁棒性优化
     - 文本清理和规范化
       - 移除标点符号和特殊字符
       - 统一数字表示格式
       - 处理同音字和近音字
     
     - 时间窗口管理
       - 动态调整识别窗口
       - 自适应噪声阈值
       - 状态超时处理
     
     - 错误恢复机制
       - 识别错误重试
       - 状态自动重置
       - 异常指令过滤

     - 缓存管理
       - 定期清理历史文本
       - 内存使用监控
       - 性能优化策略

- **性能指标**：
  - 识别延迟：<300ms
  - 采样率：16kHz
  - 识别准确率：>95%
  - 噪声容忍度：<-20dB

- **指令词表**：
  | 指令 | 触发条件 | 响应动作 |
  |-----|---------|---------|
  | "开始游戏" | 主界面 | 进入游戏状态 |
  | "退出游戏" | 任意状态 | 返回主界面 |
  | "15 15" | 游戏中 | 开始回合 |

- **异常处理**：
  - 环境噪声处理
  - 指令重复处理
  - 识别超时处理

## 3. 接口设计

### 3.1 模块间接口
```python
class HandGestureRecognizer:
    def get_hand_gesture(frame) -> dict:
        # 返回手势类型和置信度
        return {'left': 0|5, 'right': 0|5, 'confidence': float}

class VoiceRecognizer:
    def process_audio(audio_data) -> str:
        # 返回识别到的指令文本
        return command_text

class GameController:
    def process_game_state(player_gesture, robot_gesture, voice_command) -> dict:
        # 返回游戏状态更新
        return game_state
```

### 3.2 数据流设计
1. **输入流**：
   - 摄像头视频流 -> 手势识别模块
   - 麦克风音频流 -> 语音识别模块
   - 用户界面事件 -> 控制器

2. **输出流**：
   - 手势识别结果 -> 游戏逻辑
   - 语音识别结果 -> 游戏逻辑
   - 游戏状态 -> GUI显示

## 4. 性能优化设计

### 4.1 并发处理
- 使用多线程处理视频和音频输入
- GUI主线程保持响应
- 使用队列进行线程间通信

### 4.2 资源管理
- 视频帧缓存控制
- 音频缓冲区管理
- 内存使用优化

### 4.3 异常处理机制
```python
try:
    # 核心功能代码
except VideoDeviceError:
    # 摄像头异常处理
except AudioDeviceError:
    # 麦克风异常处理
except RecognitionError:
    # 识别失败处理
finally:
    # 资源释放
```

