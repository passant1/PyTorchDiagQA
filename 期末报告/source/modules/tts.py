"""
语音合成 (TTS) 模块
使用 pyttsx3 实现离线语音播报
"""
import sys


class TTS:
    """
    语音合成器
    使用 pyttsx3 离线引擎
    """

    def __init__(self):
        self.engine = None
        self.available = False
        self._init_engine()

    def _init_engine(self):
        """初始化语音引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # 设置中文语音（如果可用）
            voices = self.engine.getProperty('voices')
            for voice in voices:
                # 尝试找中文语音
                if 'chinese' in voice.name.lower() or 'zh' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            # 设置语速
            self.engine.setProperty('rate', 180)
            # 设置音量
            self.engine.setProperty('volume', 0.9)
            self.available = True
        except ImportError:
            print("[TTS] pyttsx3 未安装，语音播报不可用")
            self.available = False
        except Exception as e:
            print(f"[TTS] 语音引擎初始化失败: {e}")
            self.available = False

    def speak(self, text: str) -> bool:
        """
        朗读文本

        Args:
            text: 要朗读的文本

        Returns:
            bool: 是否成功朗读
        """
        if not self.available:
            print("[TTS] 当前环境不支持语音播报")
            return False

        if not text.strip():
            return False

        try:
            # 清理文本，移除格式字符以便朗读
            clean_text = self._clean_for_speech(text)
            self.engine.say(clean_text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS] 朗读失败: {e}")
            self.available = False
            return False

    def speak_async(self, text: str):
        """
        异步朗读（不阻塞主线程）
        注意：pyttsx3 在部分平台上异步支持有限
        """
        if not self.available:
            return

        try:
            import threading
            def _speak():
                try:
                    self.speak(text)
                except Exception:
                    pass

            t = threading.Thread(target=_speak, daemon=True)
            t.start()
        except Exception as e:
            print(f"[TTS] 异步朗读启动失败: {e}")

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """清理文本，移除不适合朗读的格式字符"""
        import re
        # 移除 Markdown 格式字符
        text = re.sub(r'[*_~`#]', '', text)
        # 移除多余换行
        text = re.sub(r'\n+', '，', text)
        # 移除代码块
        text = re.sub(r'```.*?```', '（代码省略）', text, flags=re.DOTALL)
        # 移除 URL
        text = re.sub(r'https?://\S+', '（链接省略）', text)
        return text

    def stop(self):
        """停止当前朗读"""
        if self.engine and self.available:
            try:
                self.engine.stop()
            except Exception:
                pass
