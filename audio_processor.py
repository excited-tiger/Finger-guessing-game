import sounddevice as sd
import numpy as np
import queue
import threading
from funasr import AutoModel
from typing import Optional, Callable


class AudioProcessor:

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """初始化音频处理器
        
        Args:
            callback: 识别结果回调函数，参数为识别出的文本
        """
        # 音频参数
        self.sample_rate = 16000
        self.chunk_duration = 0.3    # 减小音频块大小，提高响应速度
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # ASR 模型参数
        self.chunk_size = [0, 5, 2]    # 减小 chunk size，提高实时性
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        # 初始化 ASR 模型
        self.model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

        # 创建音频队列和缓存
        self.audio_queue = queue.Queue()
        self.cache = {}
        self.last_text = ""
        self.noise_threshold = 0.01    # 噪声阈值

        # 控制标志
        self.running = False
        self.processing_thread = None

        # 回调函数
        self.callback = callback

    def audio_callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        if status:
            print(f"音频输入状态: {status}")

        # 计算音频能量
        energy = np.mean(np.abs(indata))

        # 只有当音频能量超过阈值时才处理
        if energy > self.noise_threshold:
            self.audio_queue.put(indata.copy())

    def process_audio(self):
        """处理音频数据的线程函数"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)    # 减小超时时间
                if audio_data is None:
                    break

                # 将音频数据转换为适当的格式
                audio_data = audio_data.flatten().astype(np.float32)

                # 使用模型进行识别
                res = self.model.generate(input=audio_data,
                                          cache=self.cache,
                                          is_final=False,
                                          chunk_size=self.chunk_size,
                                          encoder_chunk_look_back=self.encoder_chunk_look_back,
                                          decoder_chunk_look_back=self.decoder_chunk_look_back)

                # 处理识别结果
                if res and res[0] and res[0]['text']:
                    text = res[0]['text'].strip()

                    # 如果文本有变化才回调
                    if text != self.last_text and len(text) > 0:
                        self.last_text = text
                        if self.callback:
                            self.callback(text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"音频处理错误: {e}")

    def extract_number(self, text: str) -> Optional[int]:
        """从识别文本中提取数字
        
        Args:
            text: 识别出的文本
            
        Returns:
            提取出的数字，如果没有找到有效数字则返回 None
        """
        # 中文数字到阿拉伯数字的映射
        cn_num = {
            '零': 0,
            '一': 1,
            '二': 2,
            '三': 3,
            '四': 4,
            '五': 5,
            '六': 6,
            '七': 7,
            '八': 8,
            '九': 9,
            '十': 10,
            '百': 100,
            '千': 1000,
            '万': 10000,
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            '两': 2    # 增加"两"的支持
        }

        try:
            # 预处理文本
            text = text.replace('零', '0').replace('两', '二')

            # 尝试直接从文本中提取数字
            for word in text:
                if word in cn_num:
                    num = cn_num[word]
                    # 只返回0-5的数字
                    if 0 <= num <= 5:
                        return num
            return None
        except:
            return None

    def start(self):
        """启动音频处理器"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

        # 启动音频输入流
        self.stream = sd.InputStream(channels=1,
                                     samplerate=self.sample_rate,
                                     blocksize=self.chunk_samples,
                                     callback=self.audio_callback,
                                     dtype=np.float32)
        self.stream.start()

    def stop(self):
        """停止音频处理器"""
        if not self.running:
            return

        self.running = False
        self.audio_queue.put(None)

        if self.processing_thread:
            self.processing_thread.join()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        self.cache.clear()
        self.last_text = ""
