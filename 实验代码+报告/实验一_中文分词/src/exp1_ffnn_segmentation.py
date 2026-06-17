from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier

from corpus_1998 import load_segmentation_sentences
from utils import DATA_DIR, OUTPUT_DIR, ensure_dirs, read_lines, set_seed, write_text


TAGS = ["B", "M", "E", "S"]
TAG2IDX = {tag: i for i, tag in enumerate(TAGS)}
IDX2TAG = {i: tag for tag, i in TAG2IDX.items()}


def words_to_chars_tags(words: List[str]) -> Tuple[List[str], List[str]]:
    chars, tags = [], []
    for word in words:
        chars.extend(list(word))
        if len(word) == 1:
            tags.append("S")
        else:
            tags.extend(["B"] + ["M"] * (len(word) - 2) + ["E"])
    return chars, tags


def load_corpus() -> Tuple[List[List[str]], List[List[str]]]:
    char_sequences, tag_sequences = [], []
    word_sequences = load_segmentation_sentences(max_sentences=1200)
    if not word_sequences:
        word_sequences = [line.split() for line in read_lines(DATA_DIR / "segmentation_corpus.txt")]
    for words in word_sequences:
        chars, tags = words_to_chars_tags(words)
        char_sequences.append(chars)
        tag_sequences.append(tags)
    return char_sequences, tag_sequences


def build_vocab(char_sequences: List[List[str]]) -> Dict[str, int]:
    counter = Counter(ch for seq in char_sequences for ch in seq)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in counter.most_common():
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def make_window_features(chars: List[str], vocab: Dict[str, int], window: int = 2) -> List[List[int]]:
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in chars]
    padded = [vocab["<PAD>"]] * window + ids + [vocab["<PAD>"]] * window
    features = []
    for i in range(window, len(padded) - window):
        features.append(padded[i - window : i + window + 1])
    return features


def restore_words(chars: List[str], tags: List[str]) -> List[str]:
    words, current = [], ""
    for ch, tag in zip(chars, tags):
        if tag == "S":
            if current:
                words.append(current)
                current = ""
            words.append(ch)
        elif tag == "B":
            if current:
                words.append(current)
            current = ch
        elif tag == "M":
            current += ch
        elif tag == "E":
            current += ch
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words


def train_with_torch(x_train: np.ndarray, y_train: np.ndarray, vocab_size: int):
    import torch
    import torch.nn as nn

    class FFNNSegmenter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5 * 32, 64),
                nn.ReLU(),
                nn.Linear(64, len(TAGS)),
            )

        def forward(self, x):
            return self.net(self.embedding(x))

    model = FFNNSegmenter()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    x_tensor = torch.tensor(x_train, dtype=torch.long)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    def predict(x: np.ndarray) -> List[str]:
        with torch.no_grad():
            pred = model(torch.tensor(x, dtype=torch.long)).argmax(dim=1).tolist()
        return [IDX2TAG[i] for i in pred]

    return "PyTorch FFNN", losses, predict


def train_with_fallback(x_train: np.ndarray, y_train: np.ndarray):
    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=42,
        learning_rate_init=0.02,
    )
    clf.fit(x_train, y_train)
    losses = [float(v) for v in getattr(clf, "loss_curve_", [])]

    def predict(x: np.ndarray) -> List[str]:
        pred = clf.predict(x)
        return [IDX2TAG[int(i)] for i in pred]

    return "sklearn MLP fallback (torch not installed)", losses, predict


def run() -> dict:
    ensure_dirs()
    set_seed(42)
    char_sequences, tag_sequences = load_corpus()
    vocab = build_vocab(char_sequences)

    x_all, y_all = [], []
    for chars, tags in zip(char_sequences, tag_sequences):
        x_all.extend(make_window_features(chars, vocab))
        y_all.extend(TAG2IDX[tag] for tag in tags)
    x_train = np.asarray(x_all, dtype=np.int64)
    y_train = np.asarray(y_all, dtype=np.int64)

    try:
        model_name, losses, predict = train_with_torch(x_train, y_train, len(vocab))
    except Exception:
        model_name, losses, predict = train_with_fallback(x_train, y_train)

    test_sentence = "中国政府重视科技教育"
    test_chars = list(test_sentence)
    test_x = np.asarray(make_window_features(test_chars, vocab), dtype=np.int64)
    pred_tags = predict(test_x)
    words = restore_words(test_chars, pred_tags)

    content = [
        "实验一：基于前馈神经网络模型实现中文分词",
        "数据集：1998 年人民日报分词标注语料 199801-199806 抽样",
        f"模型：{model_name}",
        f"训练句子数：{len(char_sequences)}",
        f"训练样本字符数：{len(y_train)}",
        f"字符表大小：{len(vocab)}",
        "训练损失：",
        ", ".join(f"{loss:.4f}" for loss in losses[:20]) + (" ..." if len(losses) > 20 else ""),
        f"最终损失：{losses[-1]:.4f}" if losses else "最终损失：无",
        f"测试句子：{test_sentence}",
        "字符标注结果：",
        " ".join(f"{ch}/{tag}" for ch, tag in zip(test_chars, pred_tags)),
        "最终分词结果：",
        " / ".join(words),
    ]
    result_path = OUTPUT_DIR / "exp1_segmentation_result.txt"
    write_text(result_path, "\n".join(content) + "\n")
    return {"name": "实验一 中文分词", "model": model_name, "loss": losses[-1] if losses else None, "result": " / ".join(words), "path": str(result_path)}


if __name__ == "__main__":
    summary = run()
    print(summary)
