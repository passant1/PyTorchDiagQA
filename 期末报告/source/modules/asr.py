"""
语音识别 (ASR) 模块
扩展模块：支持语音输入
由于依赖复杂，此模块仅保留接口，不做强制实现
"""


class ASR:
    """
    语音识别器
    使用 SpeechRecognition 库（可选扩展）
    注意：本模块为扩展功能，依赖复杂时仅保留接口
    """

    def __init__(self):
        self.available = False
        self._init_engine()

    def _init_engine(self):
        """尝试初始化语音识别"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.available = True
        except ImportError:
            print("[ASR] speech_recognition 未安装，语音输入不可用")
            self.available = False
        except OSError:
            print("[ASR] 未检测到麦克风设备，语音输入不可用")
            self.available = False
        except Exception as e:
            print(f"[ASR] 语音识别初始化失败: {e}")
            self.available = False

    def listen(self, language: str = "zh-CN") -> str:
        """
        从麦克风录音并识别

        Args:
            language: 识别语言，默认中文

        Returns:
            str: 识别到的文本，失败返回空字符串
        """
        if not self.available:
            return ""

        try:
            import speech_recognition as sr
            with self.microphone as source:
                print("[ASR] 正在聆听...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("[ASR] 正在识别...")
            text = self.recognizer.recognize_google(audio, language=language)
            return text
        except Exception as e:
            print(f"[ASR] 语音识别失败: {e}")
            return ""

    def listen_from_file(self, file_path: str, language: str = "zh-CN") -> str:
        """
        从音频文件识别

        Args:
            file_path: 音频文件路径
            language: 识别语言

        Returns:
            str: 识别到的文本
        """
        if not self.available:
            return ""

        try:
            import speech_recognition as sr
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language=language)
            return text
        except Exception as e:
            print(f"[ASR] 文件识别失败: {e}")
            return ""
