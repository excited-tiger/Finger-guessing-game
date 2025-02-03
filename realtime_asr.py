from funasr import AutoModel
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# 音频参数设置
sample_rate = 16000    # 采样率
chunk_duration = 0.6    # 每个音频块的持续时间（秒）
chunk_samples = int(sample_rate * chunk_duration)    # 每个音频块的采样点数

# 模型参数设置
chunk_size = [0, 10, 5]    # [0, 10, 5] 600ms
encoder_chunk_look_back = 4    # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1    # number of encoder chunks to lookback for decoder cross-attention

# 初始化模型
model = AutoModel(model="paraformer-zh-streaming")

# 创建音频队列
audio_queue = queue.Queue()
running = True


def audio_callback(indata, frames, time, status):
    """音频输入回调函数"""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def process_audio():
    """处理音频数据的线程函数"""
    cache = {}

    while running:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:
                break

            # 将音频数据转换为适当的格式
            audio_data = audio_data.flatten().astype(np.float32)

            # 使用模型进行识别
            res = model.generate(input=audio_data,
                                 cache=cache,
                                 is_final=False,
                                 chunk_size=chunk_size,
                                 encoder_chunk_look_back=encoder_chunk_look_back,
                                 decoder_chunk_look_back=decoder_chunk_look_back)

            if res and res[0] and res[0]['text']:    # 只有当有识别结果时才打印
                print("识别结果:", res[0]['text'])

        except queue.Empty:
            continue
        except Exception as e:
            print(f"处理错误: {e}")


def main():
    global running

    # 创建并启动音频处理线程
    process_thread = threading.Thread(target=process_audio)
    process_thread.start()

    try:
        with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=chunk_samples, callback=audio_callback):
            print("开始录音，按 Ctrl+C 停止...")
            while True:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n停止录音...")
    finally:
        running = False
        audio_queue.put(None)    # 发送停止信号
        process_thread.join()


if __name__ == "__main__":
    main()
