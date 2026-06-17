"""
意图识别与实体抽取模块
负责解析用户输入，识别意图和实体
"""
import re


# ==========================================
# 意图类型定义
# ==========================================
INTENT_CAUSE = "QUERY_CAUSE"        # 询问原因
INTENT_SOLUTION = "QUERY_SOLUTION"  # 询问解决方法
INTENT_COMMAND = "QUERY_COMMAND"    # 询问检查命令
INTENT_EXPLAIN = "QUERY_EXPLAIN"    # 询问概念或错误含义
INTENT_ALL = "QUERY_ALL"            # 综合诊断

# ==========================================
# 已知实体词典（PyTorch API、报错关键词、概念）
# ==========================================
API_PATTERNS = [
    r'torch\.cuda\.is_available',
    r'torch\.cuda\.empty_cache',
    r'torch\.cuda\.memory_summary',
    r'torch\.cuda\.synchronize',
    r'torch\.save',
    r'torch\.load',
    r'torch\.version\.cuda',
    r'torch\.nn\.CrossEntropyLoss',
    r'torch\.nn\.NLLLoss\b',
    r'torch\.utils\.data\.DataLoader',
    r'torch\.utils\.data\.Dataset',
    r'torch\.jit\.script',
    r'torch\.jit\.trace',
    r'torch\.no_grad',
    r'torch\.cuda\.amp',
    r'torch\.Tensor\.to\b',
    r'torch\.device\b',
]

ERROR_KEYWORDS = {
    "CUDA out of memory": "CUDA显存不足",
    "out of memory": "CUDA显存不足",
    "OOM": "CUDA显存不足",
    "CUDA error": "CUDA不可用",
    "no CUDA-capable device": "CUDA不可用",
    "BrokenPipeError": "DataLoader多进程问题",
    "Error(s) in loading state_dict": "state_dict不匹配",
    "Missing key(s)": "state_dict不匹配",
    "Unexpected key(s)": "state_dict不匹配",
    "state_dict mismatch": "state_dict不匹配",
    "dtype mismatch": "state_dict不匹配",
    "weights_only": "torch.load安全问题",
    "pickle": "torch.load安全问题",
    "CrossEntropyLoss": "CrossEntropyLoss输入问题",
    "target size": "CrossEntropyLoss输入问题",
    "expected scalar type": "CrossEntropyLoss输入问题",
}

CONCEPT_KEYWORDS = {
    "state_dict": "state_dict",
    "DataLoader": "DataLoader",
    "num_workers": "num_workers",
    "batch_size": "batch_size",
    "pin_memory": "pin_memory",
    "shuffle": "shuffle",
    "map_location": "map_location",
    "weights_only": "weights_only",
    "drop_last": "drop_last",
    "CrossEntropyLoss": "CrossEntropyLoss",
    "CUDA": "CUDA",
    "GPU": "GPU",
    "显存": "GPU Memory",
    "device": "torch.device",
}

PROBLEM_KEYWORDS = {
    "不可用": "CUDA不可用",
    "不能用": "CUDA不可用",
    "检测不到": "CUDA不可用",
    "返回.*False": "CUDA不可用",
    "is_available.*False": "CUDA不可用",
    "False.*怎么办": "CUDA不可用",
    "out of memory": "CUDA显存不足",
    "显存不足": "CUDA显存不足",
    "内存不足": "CUDA显存不足",
    "num_workers": "DataLoader参数问题",
    "DataLoader.*报错": "DataLoader多进程问题",
    "BrokenPipeError": "DataLoader多进程问题",
    "多进程.*问题": "DataLoader多进程问题",
    "保存.*模型": "模型保存问题",
    "加载.*模型": "模型加载问题",
    "state_dict.*不匹配": "state_dict不匹配",
    "mismatch": "state_dict不匹配",
    "weights_only": "torch.load安全问题",
    "安全.*加载": "torch.load安全问题",
    "CrossEntropyLoss": "CrossEntropyLoss输入问题",
    "交叉熵": "CrossEntropyLoss输入问题",
    "安装.*版本": "PyTorch安装版本问题",
    "CPU.*GPU": "PyTorch安装版本问题",
}

ENTITY_LABELS = {
    "CUDA不可用": "Problem",
    "CUDA显存不足": "Problem",
    "DataLoader参数问题": "Problem",
    "DataLoader多进程问题": "Problem",
    "模型保存问题": "Problem",
    "模型加载问题": "Problem",
    "state_dict不匹配": "Problem",
    "torch.load安全问题": "Problem",
    "CrossEntropyLoss输入问题": "Problem",
    "PyTorch安装版本问题": "Problem",
}


def parse_intent(text: str) -> str:
    """
    根据用户输入文本识别意图
    返回意图类型字符串
    """
    text_lower = text.lower()

    # 问原因
    cause_patterns = [r'为什么', r'原因', r'怎么会', r'为啥', r'怎么回事', r'是什么原因']
    for p in cause_patterns:
        if re.search(p, text):
            return INTENT_CAUSE

    # 问解决方法
    solution_patterns = [r'怎么办', r'怎么解决', r'如何解决', r'如何处理',
                         r'怎么修复', r'怎么弄', r'怎么处理', r'如何修复']
    for p in solution_patterns:
        if re.search(p, text):
            return INTENT_SOLUTION

    # 问检查命令
    command_patterns = [r'命令', r'怎么检查', r'如何检查', r'怎么查看',
                        r'如何查看', r'怎么看', r'命令行']
    for p in command_patterns:
        if re.search(p, text):
            return INTENT_COMMAND

    # 问概念/含义
    explain_patterns = [r'是什么', r'什么意思', r'含义', r'概念', r'介绍', r'区别']
    for p in explain_patterns:
        if re.search(p, text):
            return INTENT_EXPLAIN

    # 默认综合诊断
    return INTENT_ALL


def extract_entities(text: str) -> dict:
    """
    从用户输入中抽取实体
    返回 {"apis": [...], "errors": [...], "concepts": [...], "problems": [...], "params": [...]}
    """
    result = {
        "apis": [],
        "errors": [],
        "concepts": [],
        "problems": [],
        "params": [],
    }

    # 1. 抽取 PyTorch API
    for pattern in API_PATTERNS:
        matches = re.findall(pattern, text)
        result["apis"].extend(matches)

    # 也匹配通用的 torch.xxx 模式
    for match in re.finditer(r'torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+', text):
        api_full = match.group(0)
        if api_full not in result["apis"]:
            result["apis"].append(api_full)

    # 2. 抽取报错关键词
    for err_kw, err_label in ERROR_KEYWORDS.items():
        if err_kw.lower() in text.lower():
            if err_label not in result["errors"]:
                result["errors"].append(err_label)

    # 3. 抽取概念
    for concept_kw, concept_label in CONCEPT_KEYWORDS.items():
        if concept_kw.lower() in text.lower():
            if concept_label not in result["concepts"]:
                result["concepts"].append(concept_label)

    # 4. 抽取问题类型
    for prob_kw, prob_label in PROBLEM_KEYWORDS.items():
        if re.search(prob_kw, text, re.IGNORECASE):
            if prob_label not in result["problems"]:
                result["problems"].append(prob_label)

    # 5. 抽取参数名
    known_params = ["num_workers", "batch_size", "shuffle", "pin_memory",
                    "drop_last", "weights_only", "map_location", "device",
                    "dtype", "strict"]
    for param in known_params:
        if param.lower() in text.lower():
            result["params"].append(param)

    return result


def clean_query(text: str) -> str:
    """
    清洗用户输入，保留核心内容
    """
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^一-龥a-zA-Z0-9\s\.\-\_\:\?\(\)\[\]\{\}\=]', '', text)
    return text


def get_primary_entity(entities: dict) -> str:
    """
    从抽取的实体中确定主要实体（用于图谱查询）
    优先级：问题 > 报错 > 概念 > API
    """
    if entities["problems"]:
        return entities["problems"][0]
    if entities["errors"]:
        return entities["errors"][0]
    if entities["concepts"]:
        return entities["concepts"][0]
    if entities["apis"]:
        return entities["apis"][0]
    return ""
