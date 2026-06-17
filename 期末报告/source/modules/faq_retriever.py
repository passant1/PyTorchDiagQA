"""
FAQ 检索模块
基于 TF-IDF 和余弦相似度，继承 QA System.py 的思想
"""
import json
import logging
import os
import re
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_BASE, "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.append(_VENDOR)

import jieba
jieba.setLogLevel(logging.ERROR)

from utils.path_utils import get_data_path


class FAQRetriever:
    """
    FAQ 检索器
    使用 TF-IDF + 余弦相似度进行中文问题匹配
    """

    def __init__(self, faq_path: str = None, threshold: float = 0.15, top_k: int = 3):
        """
        初始化 FAQ 检索器

        Args:
            faq_path: FAQ JSON 文件路径，默认为 data/faq.json
            threshold: 余弦相似度阈值，低于此值触发兜底
            top_k: 返回 top-k 个匹配结果
        """
        if faq_path is None:
            faq_path = get_data_path("faq.json")

        self.threshold = threshold
        self.top_k = top_k

        # 加载 FAQ 数据
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq_data = json.load(f)

        # 提取标准问题列表
        self.questions = [item["question"] for item in self.faq_data]
        self.documents = [
            " ".join([
                item.get("question", ""),
                item.get("category", ""),
                " ".join(item.get("keywords", [])),
                item.get("answer_ref", ""),
            ])
            for item in self.faq_data
        ]

        # 构建 TF-IDF 向量器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            token_pattern=None,
            lowercase=True,
            ngram_range=(1, 2),
        )

        # 对所有标准问题进行向量化
        self.question_vectors = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def _tokenizer(text: str) -> list:
        """使用 jieba 分词，并保留 PyTorch API、参数名和英文报错短语。"""
        text = text or ""
        protected = re.findall(
            r"torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
            r"out\s+of\s+memory|"
            r"state_dict|num_workers|weights_only|map_location|"
            r"[A-Za-z_][A-Za-z0-9_]*",
            text,
            flags=re.IGNORECASE,
        )
        tokens = [t.lower().replace(" ", "_") for t in protected if t.strip()]
        for token in jieba.cut(text):
            token = token.strip().lower()
            if token:
                tokens.append(token)
        return tokens

    def search(self, query: str, top_k: int = None) -> list:
        """
        检索与用户查询最相似的 FAQ

        Args:
            query: 用户查询文本
            top_k: 返回结果数量，默认使用 self.top_k

        Returns:
            list of dict: 包含匹配的 FAQ 项、相似度分数等信息
        """
        if top_k is None:
            top_k = self.top_k

        # 向量化用户查询，tokenizer 会负责分词
        query_vector = self.vectorizer.transform([query])

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()

        # 获取 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                item = self.faq_data[idx]
                results.append({
                    "question": item["question"],
                    "answer_ref": item["answer_ref"],
                    "category": item["category"],
                    "keywords": item["keywords"],
                    "score": round(score, 4),
                    "index": int(idx),
                })

        return results

    def get_best_match(self, query: str) -> dict:
        """
        获取最佳匹配结果
        如果没有匹配项达到阈值，返回 None 标记

        Returns:
            dict or None: 最佳匹配结果，无匹配时返回 {"fallback": True, "reason": "..."}
        """
        results = self.search(query, top_k=1)
        if results:
            return results[0]
        else:
            return {"fallback": True, "reason": "no_match_above_threshold"}

    def get_debug_info(self, query: str) -> dict:
        """
        获取调试信息
        """
        results = self.search(query, top_k=self.top_k)
        return {
            "query": query,
            "tokenized": " ".join(self._tokenizer(query)),
            "matches": results,
            "faq_total": len(self.faq_data),
            "threshold": self.threshold,
        }
