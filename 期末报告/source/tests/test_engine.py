"""
测试统一问答引擎
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_engine import QAEngine


def test_cuda_not_available():
    """测试: torch.cuda.is_available 返回 False 怎么办"""
    engine = QAEngine()
    query = "torch.cuda.is_available 返回 False 怎么办？"
    result = engine.ask(query, use_llm=False)

    assert result["answer"], "应有答案"
    assert result["intent"], "应有意图识别"
    assert result["source"] in ("RAG", "RAG+LLM", "KG", "FAQ->KG", "FAQ", "LLM", "FALLBACK"), f"未知来源: {result['source']}"
    assert result["source"].startswith("RAG"), f"主链路应优先使用 RAG，当前来源: {result['source']}"

    # 检查答案包含关键信息
    answer = result["answer"]
    has_cuda = "CUDA" in answer or "cuda" in answer.lower()

    print(f"[PASS] test_cuda_not_available")
    print(f"   查询: {query}")
    print(f"   意图: {result['intent']}")
    print(f"   实体: {result['matched_entity']}")
    print(f"   来源: {result['source']}")
    print(f"   置信度: {result['confidence']:.4f}")
    print(f"   含 CUDA 关键词: {has_cuda}")
    if result.get("debug"):
        print(f"   Debug: {result['debug']}")


def test_cuda_oom():
    """测试: CUDA out of memory 怎么解决"""
    engine = QAEngine()
    query = "CUDA out of memory 怎么解决？"
    result = engine.ask(query, use_llm=False)

    assert result["answer"], "应有答案"
    assert result["source"] in ("RAG", "RAG+LLM", "KG", "FAQ->KG", "FAQ", "LLM", "FALLBACK")

    print(f"\n[PASS] test_cuda_oom")
    print(f"   查询: {query}")
    print(f"   意图: {result['intent']}")
    print(f"   实体: {result['matched_entity']}")
    print(f"   来源: {result['source']}")


def test_dataloader_num_workers():
    """测试: DataLoader 的 num_workers 是什么意思"""
    engine = QAEngine()
    query = "DataLoader 的 num_workers 是什么意思？"
    result = engine.ask(query, use_llm=False)

    assert result["answer"], "应有答案"
    print(f"\n[PASS] test_dataloader_num_workers")
    print(f"   查询: {query}")
    print(f"   意图: {result['intent']}")
    print(f"   实体: {result['matched_entity']}")
    print(f"   来源: {result['source']}")


def test_save_load_model():
    """测试: PyTorch 怎么保存和加载模型"""
    engine = QAEngine()
    query = "PyTorch 怎么保存和加载模型？"
    result = engine.ask(query, use_llm=False)

    assert result["answer"], "应有答案"
    print(f"\n[PASS] test_save_load_model")
    print(f"   查询: {query}")
    print(f"   意图: {result['intent']}")
    print(f"   实体: {result['matched_entity']}")
    print(f"   来源: {result['source']}")


def test_cross_entropy():
    """测试: CrossEntropyLoss 输入维度应该是什么"""
    engine = QAEngine()
    query = "CrossEntropyLoss 输入维度应该是什么？"
    result = engine.ask(query, use_llm=False)

    assert result["answer"], "应有答案"
    print(f"\n[PASS] test_cross_entropy")
    print(f"   查询: {query}")
    print(f"   意图: {result['intent']}")
    print(f"   实体: {result['matched_entity']}")
    print(f"   来源: {result['source']}")


if __name__ == "__main__":
    print("=" * 60)
    print("统一问答引擎测试")
    print("=" * 60)

    test_cuda_not_available()
    test_cuda_oom()
    test_dataloader_num_workers()
    test_save_load_model()
    test_cross_entropy()

    print("\n" + "=" * 60)
    print("[PASS] All engine tests passed!")
    print("=" * 60)
