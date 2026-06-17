"""
大模型 API 调用模块
基于 LLMTalk.py 的 OpenAI 兼容接口写法
"""
import os
import yaml

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

from utils.path_utils import get_base_path, get_config_path


class LLMClient:
    """
    大模型客户端
    支持 OpenAI 兼容接口，用于答案润色和兜底回答
    """

    def __init__(self, config_path: str = None):
        """
        初始化 LLM 客户端

        Args:
            config_path: config.yaml 路径
        """
        if config_path is None:
            config_path = get_config_path()

        self.config = self._load_config(config_path)
        self.llm_config = self.config.get("llm", {})

        self.api_key = self.llm_config.get("api_key", "")
        self.base_url = self.llm_config.get("base_url", "https://api.siliconflow.cn/v1")
        self.model_name = self.llm_config.get("model_name", "deepseek-ai/DeepSeek-V3")
        self.temperature = self.llm_config.get("temperature", 0.3)
        self.max_tokens = self.llm_config.get("max_tokens", 1024)
        self.enable_llm = self.llm_config.get("enable_llm", False)

        self._load_env_file()

        # 也检查环境变量，兼容作业要求和本项目早期命名
        env_api_key = (
            os.environ.get("API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("BIGMODEL_API_KEY", "")
        )
        if env_api_key:
            self.api_key = env_api_key
            self.enable_llm = True
        self.base_url = os.environ.get("BASE_URL", self.base_url)
        self.model_name = os.environ.get("MODEL_NAME") or os.environ.get("BIGMODEL_MODEL", self.model_name)

        self.client = None
        if self.api_key and self.enable_llm and HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.api_key and self.enable_llm and not HAS_OPENAI:
            print("[LLM] openai 模块未安装，大模型功能不可用。pip install openai")

    @staticmethod
    def _load_env_file():
        """轻量读取项目根目录 .env，不额外依赖 python-dotenv。"""
        env_path = os.path.join(get_base_path(), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """加载 YAML 配置文件"""
        if not os.path.exists(config_path):
            return {"llm": {}}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def is_available(self) -> bool:
        """检查大模型是否可用"""
        return self.client is not None and self.enable_llm

    def chat(self, user_message: str, system_prompt: str = None, temperature: float = None) -> str:
        """
        调用大模型对话

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数，默认使用配置值

        Returns:
            str: 模型回复内容
        """
        if not self.is_available():
            return "[大模型不可用] 请配置 API Key 或使用离线模式"

        if system_prompt is None:
            system_prompt = self._default_system_prompt()

        if temperature is None:
            temperature = self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM API 调用失败: {e}"

    @staticmethod
    def _default_system_prompt() -> str:
        """默认系统提示词：严格基于知识库上下文回答"""
        return """你是一个 PyTorch 技术助手，专门帮助用户诊断 PyTorch 报错和使用问题。

重要规则：
1. 只能根据下面提供的【知识库上下文】来回答，不能编造任何信息。
2. 不要编造 PyTorch API 名称或函数签名。
3. 不要编造 pip install 或 conda install 命令。
4. 如果知识库上下文不足以回答问题，请明确告知用户需要补充什么信息。
5. 回答时使用中文，条理清晰，分点说明。
6. 如果用户提供了完整的报错日志，请仔细分析报错原因。"""

    def polish_answer(self, raw_answer: str, context: dict = None) -> str:
        """
        对知识图谱或 FAQ 的原始答案进行语言润色

        Args:
            raw_answer: 原始结构化答案
            context: 附加上下文信息

        Returns:
            str: 润色后的答案
        """
        if not self.is_available():
            return raw_answer

        system_prompt = """你是一个 PyTorch 技术文档润色助手。
请将下面的技术答案进行语言润色，使其：
1. 更加通顺易读
2. 对于技术术语添加简短解释
3. 保持原有的结构
4. 不要添加原文没有的技术信息
5. 不要编造任何 API 或命令"""

        user_msg = f"请润色以下答案：\n\n{raw_answer}"
        if context:
            user_msg += f"\n\n附加上下文：{context}"

        polished = self.chat(user_msg, system_prompt, temperature=0.3)
        return polished

    def answer_with_rag(self, user_query: str, retrieved_context: str, intent: str = "QUERY_ALL") -> str:
        """
        基于 RAG 检索证据生成最终答案。
        检索证据作为优先参考，模型可以补充通用排查思路，但必须标明证据边界。
        """
        if not self.is_available():
            return ""

        system_prompt = """你是一个有经验的 PyTorch 报错诊断助教。
请优先参考【检索证据】回答；检索证据是本系统从 FAQ、知识图谱和官方文档摘要中召回的上下文。
你可以结合 PyTorch 通用知识补充合理的排查思路，但必须遵守：
1. 证据中明确出现的原因、命令、API 和来源要优先使用。
2. 证据没有覆盖、但属于通用经验的内容，可以写在“补充建议”中，不要伪装成检索来源。
3. 不要编造不存在的 PyTorch API、参数名、安装命令或官方文档链接。
4. 如果用户信息不足，请说明需要补充完整报错、系统环境、PyTorch 版本、CUDA/驱动版本。
回答必须使用中文，并尽量按以下结构输出：
【问题判断】
【可能原因】
【解决步骤】
【检查命令】
【补充建议】
【参考来源】"""

        user_msg = (
            f"用户问题：{user_query}\n"
            f"识别意图：{intent}\n\n"
            f"【检索证据】\n{retrieved_context}\n\n"
            "请参考上述证据给出诊断答案。证据不足的部分可以给通用排查建议，但要明确标注。"
        )
        return self.chat(user_msg, system_prompt, temperature=0.35)

    def fallback_answer(self, user_query: str, retrieved_context: str = "") -> str:
        """
        当图谱和 FAQ 都无法回答时的兜底回答

        Args:
            user_query: 用户原始问题
            retrieved_context: 检索到的有限上下文

        Returns:
            str: 兜底回答
        """
        if not self.is_available():
            return ("抱歉，系统暂未收录该问题的答案。\n"
                    "建议：\n"
                    "1. 检查 PyTorch 官方文档 https://pytorch.org/docs/\n"
                    "2. 在 PyTorch 论坛 https://discuss.pytorch.org/ 搜索相关问题\n"
                    "3. 提供完整的报错日志以便更准确地诊断")

        system_prompt = """你是 PyTorch 技术助手。用户的问题在知识库中没有直接匹配的答案。
请根据以下规则回答：
1. 如果提供的上下文中有关联信息，基于上下文谨慎回答
2. 不要编造任何 PyTorch API、函数签名或安装命令
3. 建议用户查阅官方文档或提供完整报错日志
4. 可以给出排查思路，但要明确标注这是"建议"而非"确定答案"
5. 使用中文回答"""

        user_msg = f"用户问题：{user_query}\n\n"
        if retrieved_context:
            user_msg += f"有限的检索上下文：\n{retrieved_context}\n\n"
        user_msg += "请给出谨慎的建议。"

        return self.chat(user_msg, system_prompt, temperature=0.4)
