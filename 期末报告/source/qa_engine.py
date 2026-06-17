"""
统一问答引擎
整合意图识别、FAQ 检索、知识图谱查询、大模型润色
"""
import yaml

from utils.path_utils import get_config_path
from modules.intent_parser import (
    parse_intent, extract_entities, clean_query, get_primary_entity
)
from modules.faq_retriever import FAQRetriever
from modules.kgqa import KnowledgeGraphQA
from modules.llm_client import LLMClient
from modules.rag_retriever import RAGRetriever


class QAEngine:
    """
    PyTorch 报错诊断问答引擎
    核心流程：意图识别 → 实体抽取 → 图谱查询 → FAQ 检索 → 大模型兜底
    """

    def __init__(self, config_path: str = None):
        """
        初始化问答引擎

        Args:
            config_path: config.yaml 路径
        """
        if config_path is None:
            config_path = get_config_path()

        self.config = self._load_config(config_path)

        # 初始化各模块
        self.faq_retriever = FAQRetriever(
            threshold=self.config.get("faq", {}).get("threshold", 0.15),
            top_k=self.config.get("faq", {}).get("top_k", 3),
        )
        self.kg = KnowledgeGraphQA()
        self.rag = RAGRetriever(
            threshold=self.config.get("rag", {}).get("threshold", 0.08),
            top_k=self.config.get("rag", {}).get("top_k", 5),
        )
        self.llm = LLMClient(config_path=config_path)

        self.debug_mode = self.config.get("debug", True)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """加载配置文件"""
        import os
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def ask(self, query: str, use_llm: bool = None) -> dict:
        """
        处理用户问题，返回结构化答案

        Args:
            query: 用户输入的自然语言问题
            use_llm: 是否使用大模型润色（None 则使用配置值）

        Returns:
            dict: {
                "answer": 最终答案文本,
                "intent": 识别的意图,
                "matched_entity": 匹配的图谱实体,
                "source": "RAG" | "RAG+LLM" | "KG" | "FAQ" | "FALLBACK",
                "confidence": 置信度分数,
                "debug": 调试信息,
            }
        """
        if use_llm is None:
            use_llm = self.config.get("llm", {}).get("enable_llm", False) and self.llm.is_available()

        # 1. 清洗输入
        cleaned = clean_query(query)

        # 2. 意图识别
        intent = parse_intent(cleaned)

        # 3. 实体抽取
        entities = extract_entities(cleaned)
        primary_entity = get_primary_entity(entities)

        debug_info = {
            "cleaned_query": cleaned,
            "intent": intent,
            "entities": entities,
            "primary_entity": primary_entity,
        }

        # 4. RAG 检索：主链路。KG/FAQ 只作为证据来源，不再直接决定最终回答。
        rag_query = " ".join([cleaned, primary_entity or ""])
        rag_results = self.rag.search(rag_query)
        debug_info["rag_matches"] = [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "score": r.get("score"),
                "source_doc": r.get("source_doc"),
            }
            for r in rag_results
        ]

        if rag_results:
            context = self.rag.format_context(rag_results)
            debug_info["source"] = "RAG"
            debug_info["rag_context_chars"] = len(context)

            if use_llm and self.llm.is_available():
                llm_answer = self.llm.answer_with_rag(cleaned, context, intent=intent)
                if llm_answer and not llm_answer.startswith("[大模型不可用]") and not llm_answer.startswith("LLM API"):
                    return {
                        "answer": llm_answer,
                        "intent": intent,
                        "matched_entity": primary_entity or rag_results[0].get("title", ""),
                        "source": "RAG+LLM",
                        "confidence": rag_results[0].get("score", 0.6),
                        "debug": debug_info,
                    }

            return {
                "answer": self._compose_rag_answer(cleaned, intent, primary_entity, rag_results),
                "intent": intent,
                "matched_entity": primary_entity or rag_results[0].get("title", ""),
                "source": "RAG",
                "confidence": rag_results[0].get("score", 0.6),
                "debug": debug_info,
            }

        # 5. 兼容兜底：FAQ 命中后跳转到图谱节点
        faq_result = self.faq_retriever.get_best_match(cleaned)

        if not faq_result.get("fallback"):
            # FAQ 命中，通过 answer_ref 跳转到图谱节点
            answer_ref = faq_result.get("answer_ref", "")
            debug_info["faq_match"] = faq_result
            debug_info["faq_score"] = faq_result.get("score", 0)

            if answer_ref:
                kg_result = self.kg.query(answer_ref, intent=intent)
                if kg_result.get("found"):
                    answer = self.kg.format_answer(kg_result)
                    debug_info["kg_via_faq"] = answer_ref
                    debug_info["source"] = "FAQ->KG"

                    if use_llm:
                        polished = self.llm.polish_answer(answer)
                        if polished and not polished.startswith("[大模型不可用]") and not polished.startswith("LLM API"):
                            answer = polished
                            debug_info["llm_polished"] = True

                    return {
                        "answer": answer,
                        "intent": intent,
                        "matched_entity": answer_ref,
                        "source": "FAQ->KG",
                        "confidence": faq_result.get("score", 0.5),
                        "debug": debug_info,
                    }

            # FAQ 有匹配但没有图谱跳转
            answer = f"【FAQ 匹配】\n匹配问题: {faq_result.get('question', '')}\n"
            answer += f"相关分类: {faq_result.get('category', '')}\n"
            answer += f"参考节点: {answer_ref}\n\n"
            answer += "请尝试输入更具体的 PyTorch 报错信息以获得更准确的诊断。"

            return {
                "answer": answer,
                "intent": intent,
                "matched_entity": answer_ref,
                "source": "FAQ",
                "confidence": faq_result.get("score", 0.3),
                "debug": debug_info,
            }

        # 6. 大模型兜底
        debug_info["source"] = "LLM"
        # 尝试检索一些相关上下文
        related_context = ""
        faq_results = self.faq_retriever.search(cleaned, top_k=3)
        if faq_results:
            related_context = "\n".join([
                f"相关问题: {r['question']} -> 参考: {r['answer_ref']}"
                for r in faq_results
            ])

        if use_llm and self.llm.is_available():
            fallback = self.llm.fallback_answer(cleaned, related_context)
            answer = fallback if not fallback.startswith("[大模型不可用]") and not fallback.startswith("LLM API") else self._offline_fallback()
        else:
            answer = self._offline_fallback()
            debug_info["source"] = "FALLBACK"

        return {
            "answer": answer,
            "intent": intent,
            "matched_entity": "",
            "source": debug_info["source"],
            "confidence": 0.1,
            "debug": debug_info,
        }

    @staticmethod
    def _compose_rag_answer(query: str, intent: str, entity: str, rag_results: list) -> str:
        """无 API 时，把 RAG 证据整理成可读诊断答案。"""
        kg_results = [r for r in rag_results if r.get("source") == "KnowledgeGraph"]
        faq_results = [r for r in rag_results if r.get("source") == "FAQ"]
        ordered = kg_results[:3] + faq_results[:2]
        top = ordered[0] if ordered else rag_results[0]
        lines = [
            "【RAG 检索诊断】",
            f"问题：{query}",
            f"识别主题：{entity or top.get('title', '')}",
            f"识别意图：{intent}",
            "",
            "【诊断证据】",
        ]
        for i, item in enumerate(ordered[:5], 1):
            text = item.get("text", "").strip()
            preview = text[:900] + ("..." if len(text) > 900 else "")
            lines.extend([
                f"{i}. {item.get('title', '')}（{item.get('source', '')}，相似度 {item.get('score', 0):.4f}）",
                preview,
            ])
            if item.get("url"):
                lines.append(f"来源链接：{item['url']}")
            lines.append("")

        lines.extend([
            "【下一步建议】",
            "1. 先根据上方证据中的原因和解决建议逐项排查。",
            "2. 如果仍无法解决，请补充完整报错堆栈、PyTorch 版本、Python 版本、操作系统、CUDA/驱动版本。",
            "3. 配置 API_KEY 并开启 enable_llm 后，系统会把这些证据整合成更自然的诊断回答。",
        ])
        return "\n".join(lines)

    def _offline_fallback(self) -> str:
        """离线兜底回答"""
        return (
            "抱歉，系统暂未收录该问题的答案。\n\n"
            "建议：\n"
            "1. 检查 PyTorch 官方文档 https://pytorch.org/docs/\n"
            "2. 在 PyTorch 论坛 https://discuss.pytorch.org/ 搜索相关问题\n"
            "3. 尝试输入更具体的报错信息（如完整的错误消息）\n"
            "4. 使用 nvidia-smi 检查显卡状态\n"
            "5. 使用 python -c \"import torch; print(torch.__version__)\" 检查 PyTorch 版本\n\n"
            "本系统已覆盖以下问题类型：\n"
            "- CUDA 不可用 / CUDA out of memory\n"
            "- PyTorch 安装版本问题\n"
            "- DataLoader 参数和多进程问题\n"
            "- 模型保存与加载问题\n"
            "- state_dict 不匹配\n"
            "- torch.load 安全问题\n"
            "- CrossEntropyLoss 输入问题"
        )

    def stats(self) -> dict:
        """返回系统统计信息"""
        return {
            "kg_stats": self.kg.stats(),
            "rag_stats": self.rag.stats(),
            "faq_count": len(self.faq_retriever.faq_data),
            "llm_available": self.llm.is_available(),
            "model_name": self.llm.model_name if self.llm.is_available() else "N/A",
        }


# 全局单例
_engine = None


def get_engine() -> QAEngine:
    """获取全局 QA 引擎单例"""
    global _engine
    if _engine is None:
        _engine = QAEngine()
    return _engine
