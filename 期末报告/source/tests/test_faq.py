"""
测试 FAQ 检索模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.faq_retriever import FAQRetriever


def test_faq_cuda():
    """测试 FAQ 检索能匹配 CUDA 问题"""
    retriever = FAQRetriever()

    # 测试 CUDA 相关问题
    query = "torch.cuda.is_available 返回 False 怎么办？"
    result = retriever.get_best_match(query)

    assert not result.get("fallback"), f"FAQ 应能匹配到 CUDA 问题，但触发了兜底: {result}"
    assert "CUDA" in result.get("answer_ref", "") or "cuda" in result.get("question", "").lower(), \
        f"匹配结果应包含 CUDA 相关信息: {result}"
    assert result.get("score", 0) > 0, f"相似度应大于 0: {result}"

    print(f"[PASS] test_faq_cuda")
    print(f"   查询: {query}")
    print(f"   匹配: {result['question']}")
    print(f"   图谱节点: {result['answer_ref']}")
    print(f"   相似度: {result['score']:.4f}")


def test_faq_dataloader():
    """测试 FAQ 检索能匹配 DataLoader 问题"""
    retriever = FAQRetriever()

    query = "DataLoader 的 num_workers 是什么意思？"
    result = retriever.get_best_match(query)

    assert not result.get("fallback"), f"FAQ 应能匹配: {result}"
    assert result.get("score", 0) > 0, f"相似度应大于 0"
    print(f"\n[PASS] test_faq_dataloader")
    print(f"   查询: {query}")
    print(f"   匹配: {result['question']}")
    print(f"   相似度: {result['score']:.4f}")


def test_faq_debug_info():
    """测试 Debug 信息输出"""
    retriever = FAQRetriever()
    query = "CUDA out of memory 怎么解决？"
    debug = retriever.get_debug_info(query)

    assert "query" in debug
    assert "matches" in debug
    assert len(debug["matches"]) > 0, "应有至少一个匹配结果"
    print(f"\n[PASS] test_faq_debug_info")
    print(f"   Debug 信息: {list(debug.keys())}")
    print(f"   匹配数量: {len(debug['matches'])}")


if __name__ == "__main__":
    print("=" * 60)
    print("FAQ 检索模块测试")
    print("=" * 60)

    test_faq_cuda()
    test_faq_dataloader()
    test_faq_debug_info()

    print("\n" + "=" * 60)
    print("[PASS] All FAQ tests passed!")
    print("=" * 60)
