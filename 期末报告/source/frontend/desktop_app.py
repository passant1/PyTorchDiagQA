"""
PyTorch 智能报错诊断问答系统 - 桌面 GUI
基于 Tkinter 实现，支持离线运行
"""
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_engine import get_engine
from modules.tts import TTS
from modules.asr import ASR


class DesktopApp:
    """桌面 GUI 应用程序"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PyTorch 智能报错诊断问答系统")
        self.root.geometry("900x700")
        self.root.minsize(700, 550)

        # 设置图标
        try:
            from utils.path_utils import get_asset_path
            icon_path = get_asset_path("app.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

        # 初始化引擎
        self.engine = None
        self.tts = TTS()
        self.asr = ASR()

        # 构建界面
        self._build_ui()

        # 后台初始化引擎
        self.status_var.set("正在初始化问答引擎...")
        self.root.after(100, self._init_engine)

    def _init_engine(self):
        """后台初始化引擎"""
        try:
            self.engine = get_engine()
            stats = self.engine.stats()
            kg_nodes = stats.get("kg_stats", {}).get("total_nodes", 0)
            kg_edges = stats.get("kg_stats", {}).get("total_edges", 0)
            faq_count = stats.get("faq_count", 0)
            llm_available = "可用" if stats.get("llm_available") else "离线模式"
            self.status_var.set(
                f"就绪 | 知识图谱: {kg_nodes}节点/{kg_edges}边 | FAQ: {faq_count}条 | LLM: {llm_available}"
            )
        except Exception as e:
            self.status_var.set(f"初始化失败: {e}")

    def _build_ui(self):
        """构建用户界面"""
        # ---- 顶部标题 ----
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        title_label = ttk.Label(
            title_frame,
            text="[ PyTorch 智能报错诊断问答系统 ]",
            font=("Microsoft YaHei", 16, "bold"),
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="基于文档知识图谱 | 支持 CUDA / DataLoader / 模型保存加载 / 损失函数 等问题诊断",
            font=("Microsoft YaHei", 9),
            foreground="gray",
        )
        subtitle_label.pack()

        # ---- 输入区 ----
        input_frame = ttk.LabelFrame(self.root, text="请输入您的问题或报错信息", padding=5)
        input_frame.pack(fill=tk.BOTH, padx=10, pady=5)

        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            height=4,
            font=("Consolas", 11),
            wrap=tk.WORD,
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.insert(tk.END, "torch.cuda.is_available() 返回 False 怎么办？")
        self.input_text.bind("<Control-Return>", lambda e: self._diagnose())

        # ---- 按钮区 ----
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.diagnose_btn = ttk.Button(
            btn_frame,
            text="[>] 开始诊断",
            command=self._diagnose,
        )
        self.diagnose_btn.pack(side=tk.LEFT, padx=3)

        self.build_kg_btn = ttk.Button(
            btn_frame,
            text="[KG] 构建/更新知识图谱",
            command=self._build_kg,
        )
        self.build_kg_btn.pack(side=tk.LEFT, padx=3)

        self.speak_btn = ttk.Button(
            btn_frame,
            text="[TTS] 朗读答案",
            command=self._speak_answer,
        )
        self.speak_btn.pack(side=tk.LEFT, padx=3)

        self.voice_btn = ttk.Button(
            btn_frame,
            text="[ASR] 语音输入",
            command=self._voice_input,
        )
        self.voice_btn.pack(side=tk.LEFT, padx=3)

        self.clear_btn = ttk.Button(
            btn_frame,
            text="[X] 清空",
            command=self._clear,
        )
        self.clear_btn.pack(side=tk.LEFT, padx=3)

        self.exit_btn = ttk.Button(
            btn_frame,
            text="[Q] 退出",
            command=self.root.destroy,
        )
        self.exit_btn.pack(side=tk.RIGHT, padx=3)

        # ---- 输出区 ----
        output_frame = ttk.LabelFrame(self.root, text="诊断结果", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            font=("Microsoft YaHei", 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # ---- Debug 区 ----
        debug_frame = ttk.LabelFrame(self.root, text="Debug 信息", padding=5)
        debug_frame.pack(fill=tk.BOTH, padx=10, pady=5)

        self.debug_text = scrolledtext.ScrolledText(
            debug_frame,
            height=5,
            font=("Consolas", 9),
            foreground="gray",
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.debug_text.pack(fill=tk.BOTH)

        # ---- 状态栏 ----
        self.status_var = tk.StringVar(value="正在初始化...")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            font=("Microsoft YaHei", 9),
            padding=3,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _set_output(self, text: str):
        """设置输出框内容"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)

    def _set_debug(self, text: str):
        """设置 Debug 框内容"""
        self.debug_text.config(state=tk.NORMAL)
        self.debug_text.delete(1.0, tk.END)
        self.debug_text.insert(tk.END, text)
        self.debug_text.config(state=tk.DISABLED)

    def _diagnose(self):
        """执行诊断"""
        query = self.input_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("提示", "请输入问题或报错信息")
            return

        self.status_var.set("正在诊断...")
        self.diagnose_btn.config(state=tk.DISABLED)
        self._set_output("[...] 正在分析您的问题，请稍候...")
        self._set_debug("")

        def _run():
            try:
                if self.engine is None:
                    self.engine = get_engine()
                result = self.engine.ask(query)
                self.root.after(0, lambda: self._show_result(result))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _show_result(self, result: dict):
        """显示诊断结果"""
        answer = result.get("answer", "")
        intent = result.get("intent", "")
        matched = result.get("matched_entity", "")
        source = result.get("source", "")
        confidence = result.get("confidence", 0)
        debug = result.get("debug", {})

        self._set_output(answer)

        # Debug 信息
        debug_lines = []
        debug_lines.append(f"意图: {intent}")
        debug_lines.append(f"匹配实体: {matched}")
        debug_lines.append(f"数据来源: {source}")
        debug_lines.append(f"置信度: {confidence:.4f}")
        if debug.get("faq_score"):
            debug_lines.append(f"FAQ 相似度: {debug['faq_score']:.4f}")
        if debug.get("kg_match"):
            debug_lines.append(f"图谱匹配: {debug['kg_match']}")
        if debug.get("kg_via_faq"):
            debug_lines.append(f"FAQ→图谱 跳转: {debug['kg_via_faq']}")
        if debug.get("llm_polished"):
            debug_lines.append("已使用大模型润色")
        if debug.get("entities"):
            entities = debug["entities"]
            if entities.get("apis"):
                debug_lines.append(f"识别 API: {entities['apis']}")
            if entities.get("problems"):
                debug_lines.append(f"识别问题: {entities['problems']}")
            if entities.get("errors"):
                debug_lines.append(f"识别错误: {entities['errors']}")
            if entities.get("concepts"):
                debug_lines.append(f"识别概念: {entities['concepts']}")

        self._set_debug("\n".join(debug_lines))
        self.status_var.set(f"诊断完成 | 来源: {source} | 置信度: {confidence:.2f}")
        self.diagnose_btn.config(state=tk.NORMAL)

    def _show_error(self, error_msg: str):
        """显示错误"""
        self._set_output(f"[ERROR] 诊断失败:\n{error_msg}")
        self._set_debug(f"错误: {error_msg}")
        self.status_var.set("诊断失败")
        self.diagnose_btn.config(state=tk.NORMAL)
        messagebox.showerror("错误", f"诊断过程出错:\n{error_msg}")

    def _build_kg(self):
        """构建知识图谱"""
        self.status_var.set("正在构建知识图谱...")
        self.build_kg_btn.config(state=tk.DISABLED)
        self._set_output("[...] 正在构建知识图谱，请稍候...")

        def _run():
            try:
                from kg_builder.graph_builder import build_graph
                from kg_builder.export_csv import export_to_csv

                graph = build_graph()
                export_csv = export_csv()

                # 重新加载引擎
                global _engine
                _engine = None
                self.engine = get_engine()

                msg = f"[OK] 知识图谱构建完成!\n\n节点数: {len(graph['nodes'])}\n边数: {len(graph['edges'])}\n\n"
                msg += "文件已生成:\n"
                msg += "  • data/pytorch_kg.json\n"
                msg += "  • data/nodes.csv\n"
                msg += "  • data/edges.csv"

                self.root.after(0, lambda: self._set_output(msg))
                self.root.after(0, lambda: self.status_var.set(
                    f"就绪 | 知识图谱: {len(graph['nodes'])}节点/{len(graph['edges'])}边"
                ))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"构建失败: {e}"))
            finally:
                self.root.after(0, lambda: self.build_kg_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _speak_answer(self):
        """朗读当前答案"""
        answer = self.output_text.get(1.0, tk.END).strip()
        if not answer:
            messagebox.showinfo("提示", "请先执行诊断")
            return

        if not self.tts.available:
            messagebox.showwarning("语音播报", "当前环境不支持语音播报。\n请确保已安装 pyttsx3 并配置了语音引擎。")
            return

        self.status_var.set("正在朗读...")
        self.speak_btn.config(state=tk.DISABLED)

        def _run():
            success = self.tts.speak(answer)
            if success:
                self.root.after(0, lambda: self.status_var.set("朗读完成"))
            else:
                self.root.after(0, lambda: self.status_var.set("朗读失败"))
            self.root.after(0, lambda: self.speak_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _voice_input(self):
        """语音输入"""
        if not self.asr.available:
            messagebox.showwarning(
                "语音输入",
                "语音输入功能不可用。\n"
                "请确保已安装 speech_recognition 和 PyAudio。\n"
                "安装命令: pip install SpeechRecognition pyaudio"
            )
            return

        self.status_var.set("正在聆听...")
        self.voice_btn.config(state=tk.DISABLED)

        def _run():
            text = self.asr.listen()
            if text:
                self.root.after(0, lambda: self.input_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.input_text.insert(tk.END, text))
                self.root.after(0, lambda: self.status_var.set("语音识别完成"))
            else:
                self.root.after(0, lambda: self.status_var.set("语音识别失败或无输入"))
            self.root.after(0, lambda: self.voice_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _clear(self):
        """清空所有输入输出"""
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        self._set_debug("")
        self.status_var.set("已清空")

    def run(self):
        """启动 GUI 主循环"""
        self.root.mainloop()


def main():
    """桌面应用入口"""
    app = DesktopApp()
    app.run()


if __name__ == "__main__":
    main()
