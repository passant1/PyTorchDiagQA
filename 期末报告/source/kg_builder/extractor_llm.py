"""
大模型辅助抽取器（可选模块）
使用 LLM 从文档中抽取三元组
"""
import json


def build_extraction_prompt(text_chunk: str) -> str:
    """
    构建抽取提示词

    Args:
        text_chunk: 文档文本块

    Returns:
        str: 提示词
    """
    return f"""从以下 PyTorch 文档文本中抽取知识三元组 (主体, 关系, 客体)。

支持的实体类型: Problem(问题), Error(错误), API(PyTorch API), Concept(概念), Cause(原因), Solution(解决方法), Command(命令), DocPage(文档来源), CodeExample(代码示例)

支持的关系类型: HAS_API, HAS_CAUSE, HAS_SOLUTION, CHECK_BY, MENTIONED_IN, HAS_PARAMETER, RELATED_TO, HAS_EXAMPLE, SIMILAR_TO, HAS_ERROR

请输出 JSON 数组，每个元素格式:
{{"head": "实体名", "head_type": "实体类型", "relation": "关系", "tail": "客体名", "tail_type": "客体类型", "evidence": "原文证据"}}

文档文本:
{text_chunk}

请严格只输出 JSON 数组，不要包含其他文字。"""


def extract_with_llm(text_chunk: str, llm_client) -> list:
    """
    使用大模型从文本块中抽取三元组

    Args:
        text_chunk: 文档文本块
        llm_client: LLMClient 实例

    Returns:
        list of dict: 抽取的三元组
    """
    if llm_client is None or not llm_client.is_available():
        return []

    prompt = build_extraction_prompt(text_chunk)

    system_prompt = ("你是一个 NLP 知识图谱构建助手。"
                     "你的任务是严格从给定文本中抽取结构化的知识三元组。"
                     "只输出 JSON 数组，不要添加解释。")

    response = llm_client.chat(prompt, system_prompt, temperature=0.1)

    # 尝试解析 JSON
    try:
        # 清理响应，去掉可能的 markdown 标记
        response = response.strip()
        if response.startswith("```"):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        triples = json.loads(response)
        return triples if isinstance(triples, list) else []
    except json.JSONDecodeError:
        print("[LLM Extractor] JSON 解析失败，跳过此块")
        return []
