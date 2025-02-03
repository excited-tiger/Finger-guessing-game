# 十五二十猜拳游戏

这是一个基于 PyQt6 开发的智能猜拳游戏，实现了传统的十五二十猜拳游戏的人机对战。系统集成了语音识别、实时手势识别和自动判定等功能。

## 功能特点

- 实时手势识别（支持 0-5 的数字手势）
- 语音指令识别（支持开始游戏和数字输入）
- 美观的图形界面
- 实时状态显示和游戏统计
- 支持全屏显示

## 系统要求

- Python 3.9 或更高版本
- 摄像头
- 麦克风
- 操作系统：Windows/macOS/Linux

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/hand_landmark.git
cd hand_landmark
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行游戏

```bash
python main.py
```

## 游戏规则

1. 每局游戏进行5轮有效比赛（平局不计入）
2. 玩家可以通过手势或语音输入数字（0-5）
3. 电脑随机生成数字（0-5）
4. 如果双方数字之和为15或20，且玩家猜对，玩家获胜
5. 否则电脑获胜

## 操作说明

- 开始游戏：点击"开始游戏"按钮或说"开始游戏"
- 输入数字：使用手势（0-5）或语音输入数字
- 结束游戏：点击"结束游戏"按钮或说"结束游戏"
- 切换全屏：按 ESC 键

## 注意事项

1. 确保摄像头和麦克风正常工作
2. 保持适当的光线条件以提高手势识别准确率
3. 在较安静的环境中使用语音功能
4. 手势要清晰，保持在摄像头视野范围内

## 技术栈

- PyQt6：GUI框架
- MediaPipe：手势识别
- FunASR：语音识别
- OpenCV：图像处理
- NumPy：数值计算

## 许可证

MIT License # Finger-guessing-game
