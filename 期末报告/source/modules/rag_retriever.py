"""
RAG 检索模块
将官方文档摘要、FAQ、知识图谱节点统一成可检索的证据块。
"""
import json
import logging
import os
import re
import sys
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_BASE, "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.append(_VENDOR)

import jieba

from utils.path_utils import get_data_path

jieba.setLogLevel(logging.ERROR)


class RAGRetriever:
    """轻量级 RAG 检索器，优先服务问答质量而不是图谱展示。"""

    def __init__(self, chunks_path: str = None, threshold: float = 0.08, top_k: int = 5):
        self.chunks_path = chunks_path or get_data_path("doc_chunks.json")
        self.threshold = threshold
        self.top_k = top_k
        self.chunks = self._load_chunks()
        self.documents = [self._chunk_to_document(chunk) for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            token_pattern=None,
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.vectors = self.vectorizer.fit_transform(self.documents) if self.documents else None

    def _load_chunks(self) -> List[Dict]:
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("chunks", data) if isinstance(data, dict) else data
        return self._build_chunks_from_local_data()

    def _build_chunks_from_local_data(self) -> List[Dict]:
        """doc_chunks.json 缺失时，从 KG 和 FAQ 动态生成证据块。"""
        chunks = []
        kg_path = get_data_path("pytorch_kg.json")
        if os.path.exists(kg_path):
            with open(kg_path, "r", encoding="utf-8") as f:
                kg = json.load(f)
            nodes = {node.get("id"): node for node in kg.get("nodes", [])}
            outgoing = {}
            for edge in kg.get("edges", []):
                outgoing.setdefault(edge.get("source"), []).append(edge)
            for node in kg.get("nodes", []):
                if node.get("label") != "Problem":
                    continue
                parts = [node.get("description", "")]
                for edge in outgoing.get(node.get("id"), []):
                    target = nodes.get(edge.get("target"), {})
                    if target:
                        parts.append(f"{edge.get('relation')}: {target.get('name')}。{target.get('description', '')}")
                chunks.append({
                    "id": f"kg_{node.get('id')}",
                    "title": node.get("name", ""),
                    "category": node.get("name", ""),
                    "source": "KnowledgeGraph",
                    "source_doc": "data/pytorch_kg.json",
                    "url": "",
                    "text": "\n".join(p for p in parts if p),
                    "keywords": node.get("aliases", []) + [node.get("name", "")],
                })

        faq_path = get_data_path("faq.json")
        if os.path.exists(faq_path):
            with open(faq_path, "r", encoding="utf-8") as f:
                faq = json.load(f)
            for idx, item in enumerate(faq):
                chunks.append({
                    "id": f"faq_{idx}",
                    "title": item.get("question", ""),
                    "category": item.get("category", ""),
                    "source": "FAQ",
                    "source_doc": "data/faq.json",
                    "url": "",
                    "text": f"常见问题：{item.get('question', '')}\n对应诊断主题：{item.get('answer_ref', '')}",
                    "keywords": item.get("keywords", []),
                })
        return chunks

    @staticmethod
    def _tokenizer(text: str) -> List[str]:
        text = text or ""
        protected = re.findall(
            r"torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
            r"out\s+of\s+memory|"
            r"state_dict|num_workers|weights_only|map_location|"
            r"CrossEntropyLoss|DataLoader|CUDA|GPU|OOM|"
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

    @staticmethod
    def _chunk_to_document(chunk: Dict) -> str:
        return "\n".join([
            chunk.get("title", ""),
            chunk.get("category", ""),
            chunk.get("text", ""),
            " ".join(chunk.get("keywords", [])),
        ])

    def search(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict]:
        if not self.chunks or self.vectors is None:
            return []
        top_k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors).flatten()
        query_lower = (query or "").lower()
        ranked = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[int(idx)]
            adjusted = float(score)
            title = chunk.get("title", "").lower()
            category = chunk.get("category", "").lower()
            if chunk.get("source") == "KnowledgeGraph":
                adjusted += 0.04
            if title and title in query_lower:
                adjusted += 0.12
            if category and category in query_lower:
                adjusted += 0.08
            ranked.append((adjusted, idx, float(score)))
        indices = sorted(ranked, reverse=True)[:top_k]
        results = []
        for adjusted, idx, raw_score in indices:
            if adjusted < threshold:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = round(adjusted, 4)
            chunk["raw_score"] = round(raw_score, 4)
            results.append(chunk)
        return results

    @staticmethod
    def format_context(results: List[Dict], max_chars: int = 5000) -> str:
        blocks = []
        total = 0
        for i, item in enumerate(results, 1):
            block = (
                f"[证据{i}] {item.get('title', '')}\n"
                f"来源: {item.get('source_doc') or item.get('source', '')}\n"
                f"链接: {item.get('url', '')}\n"
                f"内容: {item.get('text', '')}\n"
            )
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n".join(blocks)

    def stats(self) -> Dict:
        return {
            "chunk_count": len(self.chunks),
            "threshold": self.threshold,
            "top_k": self.top_k,
        }
