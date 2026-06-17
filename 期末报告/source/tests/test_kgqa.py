"""
测试知识图谱问答模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.kgqa import KnowledgeGraphQA


def test_kg_exists():
    """测试知识图谱是否能加载"""
    kg = KnowledgeGraphQA()
    stats = kg.stats()

    assert stats["total_nodes"] > 0, "知识图谱应有节点"
    assert stats["total_edges"] > 0, "知识图谱应有边"

    print(f"[PASS] test_kg_exists")
    print(f"   节点数: {stats['total_nodes']}")
    print(f"   边数: {stats['total_edges']}")
    print(f"   节点类型: {stats['node_types']}")


def test_query_cuda_not_available():
    """测试查询 CUDA 不可用"""
    kg = KnowledgeGraphQA()
    result = kg.query("CUDA不可用", intent="QUERY_SOLUTION")

    assert result["found"], f"应能找到 CUDA不可用: {result}"
    assert len(result.get("causes", [])) > 0, "应有原因"
    assert len(result.get("solutions", [])) > 0, "应有解决方法"
    assert len(result.get("apis", [])) > 0, "应有相关 API"

    print(f"\n[PASS] test_query_cuda_not_available")
    print(f"   节点: {result['entity']['name']}")
    print(f"   原因数: {len(result['causes'])}")
    print(f"   解决方法数: {len(result['solutions'])}")
    print(f"   API 数: {len(result['apis'])}")


def test_query_state_dict_mismatch():
    """测试查询 state_dict 不匹配"""
    kg = KnowledgeGraphQA()
    result = kg.query("state_dict不匹配", intent="QUERY_ALL")

    assert result["found"], f"应能找到 state_dict不匹配: {result}"
    assert len(result.get("solutions", [])) > 0, "应有解决方法"
    assert len(result.get("causes", [])) > 0, "应有原因"

    print(f"\n[PASS] test_query_state_dict_mismatch")
    print(f"   节点: {result['entity']['name']}")
    print(f"   原因数: {len(result['causes'])}")
    print(f"   解决方法数: {len(result['solutions'])}")


def test_format_answer():
    """测试答案格式化"""
    kg = KnowledgeGraphQA()
    result = kg.query("CUDA显存不足", intent="QUERY_ALL")
    answer = kg.format_answer(result)

    assert "【问题类型】" in answer or "CUDA显存不足" in answer
    assert "【可能原因】" in answer
    assert "【解决建议】" in answer
    assert "【检查命令】" in answer or "【相关 API】" in answer

    print(f"\n[PASS] test_format_answer")
    print(f"   答案长度: {len(answer)} 字符")


def test_stats():
    """测试图谱统计"""
    kg = KnowledgeGraphQA()
    stats = kg.stats()

    assert stats["total_nodes"] >= 40, f"节点数应 >= 40，当前: {stats['total_nodes']}"
    assert stats["total_edges"] >= 80, f"边数应 >= 80，当前: {stats['total_edges']}"

    print(f"\n[PASS] test_stats")
    print(f"   节点数: {stats['total_nodes']} (要求 >= 40)")
    print(f"   边数: {stats['total_edges']} (要求 >= 80)")


if __name__ == "__main__":
    print("=" * 60)
    print("知识图谱问答模块测试")
    print("=" * 60)

    test_kg_exists()
    test_query_cuda_not_available()
    test_query_state_dict_mismatch()
    test_format_answer()
    test_stats()

    print("\n" + "=" * 60)
    print("[PASS] All KGQA tests passed!")
    print("=" * 60)
