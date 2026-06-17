from __future__ import annotations

import itertools
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from corpus_1998 import load_similarity_sentences
from utils import DATA_DIR, OUTPUT_DIR, ensure_dirs, read_lines, try_jieba_cut, write_text, format_matrix


def tokenize_sentences(sentences: List[str]) -> List[List[str]]:
    return [try_jieba_cut(sentence) for sentence in sentences]


def run() -> dict:
    ensure_dirs()
    sentences = load_similarity_sentences(max_sentences=6)
    if not sentences:
        sentences = read_lines(DATA_DIR / "similarity_examples.txt")
    tokenized = tokenize_sentences(sentences)
    documents = [" ".join(tokens) for tokens in tokenized]

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectors = vectorizer.fit_transform(documents)
    sim_matrix = cosine_similarity(vectors)
    feature_names = vectorizer.get_feature_names_out().tolist()

    labels = [f"S{i+1}" for i in range(len(sentences))]
    pairs = []
    for i, j in itertools.combinations(range(len(sentences)), 2):
        pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    lines = [
        "实验二：词向量表示以及句子相似度计算",
        "数据集：1998 年人民日报分词标注语料 199801-199806 抽样句子",
        "方法：TF-IDF 句向量 + 余弦相似度",
        "原始句子：",
    ]
    lines.extend(f"{labels[i]}: {sentence}" for i, sentence in enumerate(sentences))
    lines.append("\n分词结果：")
    lines.extend(f"{labels[i]}: {' / '.join(tokens)}" for i, tokens in enumerate(tokenized))
    lines.append("\nTF-IDF 特征词：")
    lines.append(", ".join(feature_names))
    lines.append(f"\n句子向量维度：{vectors.shape[1]}")
    lines.append("\n句子相似度矩阵：")
    lines.append(format_matrix(sim_matrix, labels=labels, digits=3))
    lines.append("\n相似度解释：")
    for score, i, j in pairs[:4]:
        lines.append(f"{labels[i]} 与 {labels[j]} 相似度 {score:.3f}：主题或词汇重合程度较高。")
    for score, i, j in pairs[-2:]:
        lines.append(f"{labels[i]} 与 {labels[j]} 相似度 {score:.3f}：主题差异较大或共享词较少。")

    result_path = OUTPUT_DIR / "exp2_similarity_result.txt"
    write_text(result_path, "\n".join(lines) + "\n")
    return {
        "name": "实验二 句子相似度",
        "features": len(feature_names),
        "top_pair": f"S{pairs[0][1]+1}-S{pairs[0][2]+1}",
        "top_score": float(pairs[0][0]),
        "path": str(result_path),
    }


if __name__ == "__main__":
    print(run())
