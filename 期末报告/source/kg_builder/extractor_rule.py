"""
规则抽取器
使用正则表达式从文档文本中抽取实体和关系
"""
import re


def extract_apis(text: str) -> list:
    """
    抽取 PyTorch API

    Returns:
        list of str: API 名称列表
    """
    apis = set()

    # 完整 API 模式: torch.xxx.yyy
    pattern = r'torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+'
    matches = re.findall(pattern, text)
    apis.update(matches)

    # 过滤太短或太通用的
    apis = {a for a in apis if len(a) > 7 and a.count('.') >= 1}

    return sorted(apis)


def extract_commands(text: str) -> list:
    """
    抽取命令行命令

    Returns:
        list of dict: [{"command": "...", "description": "..."}]
    """
    commands = []

    # pip install
    pip_pattern = r'(pip\s+(?:install|uninstall)\s+[^\n]+)'
    for match in re.finditer(pip_pattern, text, re.IGNORECASE):
        commands.append({"command": match.group(1).strip(), "type": "pip"})

    # conda install
    conda_pattern = r'(conda\s+(?:install|create)\s+[^\n]+)'
    for match in re.finditer(conda_pattern, text, re.IGNORECASE):
        commands.append({"command": match.group(1).strip(), "type": "conda"})

    # python -c
    py_pattern = r'(python\s+-c\s+"[^"]+")'
    for match in re.finditer(py_pattern, text):
        commands.append({"command": match.group(1).strip(), "type": "python"})

    # nvidia-smi
    if 'nvidia-smi' in text.lower():
        commands.append({"command": "nvidia-smi", "type": "system"})

    return commands


def extract_code_blocks(text: str) -> list:
    """
    抽取代码块

    Returns:
        list of dict: [{"code": "...", "description": "..."}]
    """
    code_blocks = []

    # Markdown 风格代码块
    md_pattern = r'```(?:python)?\s*\n(.*?)```'
    for match in re.finditer(md_pattern, text, re.DOTALL):
        code = match.group(1).strip()
        if any(kw in code for kw in ["import torch", "torch.save", "torch.load",
                                       "DataLoader", "CrossEntropyLoss"]):
            code_blocks.append({"code": code, "type": "python"})

    # 内联代码
    inline_pattern = r'`([^`]+)`'
    for match in re.finditer(inline_pattern, text):
        code_text = match.group(1)
        if any(kw in code_text for kw in ["torch.", "import ", "pip ", "python "]):
            code_blocks.append({"code": code_text, "type": "inline"})

    return code_blocks


def extract_parameters(text: str) -> list:
    """
    抽取参数名

    Returns:
        list of str: 参数名列表
    """
    known_params = [
        "num_workers", "batch_size", "shuffle", "pin_memory",
        "drop_last", "weights_only", "map_location", "device",
        "dtype", "strict", "collate_fn", "persistent_workers"
    ]
    found = []
    for param in known_params:
        if param in text:
            found.append(param)
    return found


def extract_problem_type(text: str) -> list:
    """
    根据关键词推断问题类型

    Returns:
        list of str: 问题类型列表
    """
    problems = set()

    rules = [
        (r'cuda.*(?:不可用|不能用|不可用|not\s+available|is_available.*False)', "CUDA不可用"),
        (r'(?:out\s+of\s+memory|OOM|memory\s+error)', "CUDA显存不足"),
        (r'DataLoader.*num_workers', "DataLoader参数问题"),
        (r'(?:BrokenPipeError|multiprocessing|多进程)', "DataLoader多进程问题"),
        (r'(?:save|保存).*(?:model|模型)', "模型保存问题"),
        (r'(?:load|加载).*(?:model|模型)', "模型加载问题"),
        (r'state_dict.*(?:mismatch|不匹配|missing|unexpected)', "state_dict不匹配"),
        (r'weights_only|pickle.*(?:安全|attack)', "torch.load安全问题"),
        (r'CrossEntropyLoss|target.*dimension|target.*size|logits.*shape', "CrossEntropyLoss输入问题"),
        (r'pip.*install.*torch|CPU.*版本|GPU.*版本', "PyTorch安装版本问题"),
    ]

    for pattern, problem_type in rules:
        if re.search(pattern, text, re.IGNORECASE):
            problems.add(problem_type)

    return list(problems)
