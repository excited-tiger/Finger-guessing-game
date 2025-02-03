import os
import platform
import shutil


def setup_chinese_font():
    """设置中文字体"""
    # 目标字体文件路径
    target_font = "HONORSansCN-Regular.ttf"

    # 指定 HONOR Sans 字体路径
    honor_font = "/Users/sunzhenping/Downloads/HONOR_Sans_1.2/HONORSansCN/HONORSansCN-Regular.ttf"

    # 如果字体文件已存在，直接返回
    if os.path.exists(target_font):
        print("字体文件已存在")
        return True

    # 复制 HONOR Sans 字体
    if os.path.exists(honor_font):
        try:
            shutil.copy2(honor_font, target_font)
            print(f"已复制字体文件: {honor_font} -> {target_font}")
            return True
        except Exception as e:
            print(f"复制字体文件失败: {e}")
            return False
    else:
        print("未找到 HONOR Sans 字体文件")
        return False


if __name__ == "__main__":
    setup_chinese_font()
