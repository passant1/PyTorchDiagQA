"""
测试 RAG 检索模块。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rag_retriever import RAGRetriever


def test_rag_loads_chunks():
    retriever = RAGRetriever()
    stats = retriever.stats()
    assert stats["chunk_count"] >= 30


def test_rag_cuda_query():
    retriever = RAGRetriever()
    results = retriever.search("torch.cuda.is_available 返回 False 怎么办？")
    assert results
    joined = "\n".join(r["title"] + r["text"] for r in results[:3])
    assert "CUDA" in joined or "cuda" in joined.lower()


def test_rag_dataloader_query():
    retriever = RAGRetriever()
    results = retriever.search("DataLoader 的 num_workers 是什么意思？")
    assert results
    joined = "\n".join(r["title"] + r["text"] for r in results[:3])
    assert "num_workers" in joined or "DataLoader" in joined
