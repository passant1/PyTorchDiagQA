from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def simple_tokenize(text: str) -> List[str]:
    parts = text.strip().split()
    if len(parts) > 1:
        return parts
    return [ch for ch in text.strip() if ch.strip()]


def try_jieba_cut(text: str) -> List[str]:
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        return simple_tokenize(text)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_matrix(matrix: Iterable[Iterable[float]], labels: List[str] | None = None, digits: int = 3) -> str:
    arr = np.asarray(matrix)
    if labels:
        header = " " * 12 + "".join(f"{label:>12}" for label in labels)
        rows = [header]
        for label, row in zip(labels, arr):
            rows.append(f"{label:<12}" + "".join(f"{v:>12.{digits}f}" for v in row))
        return "\n".join(rows)
    return "\n".join(" ".join(f"{v:.{digits}f}" for v in row) for row in arr)
