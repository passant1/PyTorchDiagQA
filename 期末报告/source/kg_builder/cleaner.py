"""
文本清洗模块
对抓取的文档文本进行清洗和分块
"""
import re


def clean_text(text: str) -> str:
    """
    清洗文本

    Args:
        text: 原始文本

    Returns:
        str: 清洗后的文本
    """
    # 压缩空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 压缩空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除每行首尾空白
    lines = [line.strip() for line in text.split('\n')]
    # 去除空行
    lines = [l for l in lines if l]
    return '\n'.join(lines)


def split_into_chunks(text: str, max_chars: int = 2000, overlap: int = 200) -> list:
    """
    将文本切分为多个块

    Args:
        text: 输入文本
        max_chars: 每块最大字符数
        overlap: 块间重叠字符数

    Returns:
        list of str: 文本块列表
    """
    paragraphs = text.split('\n')
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + '\n'
        else:
            if current:
                chunks.append(current.strip())
            current = para + '\n'
            # 处理超长段落
            while len(current) > max_chars:
                chunks.append(current[:max_chars].strip())
                current = current[max_chars - overlap:]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def extract_sections(text: str) -> list:
    """
    按标题将文本拆分为章节

    Returns:
        list of (title, content) tuples
    """
    sections = []
    lines = text.split('\n')
    current_title = "概述"
    current_lines = []

    for line in lines:
        # 检测标题行（以 # 开头或全大写）
        if re.match(r'^#{1,3}\s', line) or (line.isupper() and len(line) > 3):
            if current_lines:
                sections.append((current_title, '\n'.join(current_lines)))
            current_title = line.lstrip('#').strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, '\n'.join(current_lines)))

    return sections
