from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from utils import PROJECT_ROOT


LOCAL_CORPUS_DIR = PROJECT_ROOT / "199801"
PARENT_CORPUS_DIR = PROJECT_ROOT.parent / "199801"
CORPUS_DIR = LOCAL_CORPUS_DIR if LOCAL_CORPUS_DIR.exists() else PARENT_CORPUS_DIR
CORPUS_FILES = [CORPUS_DIR / f"19980{i}.txt" for i in range(1, 7)]
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_people_daily_line(line: str) -> List[str]:
    words: List[str] = []
    parts = line.strip().split()
    for token in parts[1:]:
        token = token.strip()
        if "/" not in token:
            continue
        word, pos = token.rsplit("/", 1)
        word = word.lstrip("[")
        if not word or pos.startswith("w") or PUNCT_RE.fullmatch(word):
            continue
        words.append(word)
    return words


def iter_people_daily_sentences(files: Iterable[Path] = CORPUS_FILES) -> Iterable[List[str]]:
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            words = parse_people_daily_line(line)
            if words:
                yield words


def load_segmentation_sentences(max_sentences: int = 1200, max_chars: int = 80) -> List[List[str]]:
    sentences: List[List[str]] = []
    for words in iter_people_daily_sentences():
        if 4 <= sum(len(w) for w in words) <= max_chars:
            sentences.append(words)
        if len(sentences) >= max_sentences:
            break
    return sentences


def load_similarity_sentences(max_sentences: int = 6) -> List[str]:
    picked: List[str] = []
    preferred = ["科技", "教育", "学校", "经济", "体育", "比赛", "发展", "研究"]
    for words in iter_people_daily_sentences():
        sentence = "".join(words)
        if 12 <= len(sentence) <= 45 and any(key in sentence for key in preferred):
            picked.append(" ".join(words))
        if len(picked) >= max_sentences:
            break
    return picked


def load_keyword_classification(max_per_class: int = 60) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    keywords = {
        "体育": ["比赛", "球队", "运动员", "足球", "篮球", "冠军", "联赛", "体育", "教练"],
        "科技": ["科技", "计算机", "网络", "软件", "技术", "信息", "科学", "电子", "研究"],
        "教育": ["教育", "学校", "学生", "教师", "大学", "课程", "教学", "高校", "考试"],
    }
    texts: List[str] = []
    labels: List[str] = []
    examples: Dict[str, List[str]] = {label: [] for label in keywords}
    counts = {label: 0 for label in keywords}

    for words in iter_people_daily_sentences():
        sentence = "".join(words)
        if not 12 <= len(sentence) <= 80:
            continue
        for label, keys in keywords.items():
            if counts[label] >= max_per_class:
                continue
            if any(key in sentence for key in keys):
                texts.append(sentence)
                labels.append(label)
                counts[label] += 1
                if len(examples[label]) < 3:
                    examples[label].append(sentence)
                break
        if all(count >= max_per_class for count in counts.values()):
            break
    return texts, labels, examples


def corpus_summary() -> Dict[str, int | str]:
    existing = [path for path in CORPUS_FILES if path.exists()]
    line_count = 0
    token_count = 0
    for path in existing:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line_count += 1
            token_count += len(parse_people_daily_line(line))
    return {
        "path": str(CORPUS_DIR),
        "files": len(existing),
        "lines": line_count,
        "tokens": token_count,
    }
