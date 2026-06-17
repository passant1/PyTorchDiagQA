# NLP Homework — 实验代码与期末大作业

自然语言处理课程的实验代码与期末大作业（PyTorch 智能报错诊断问答系统）。

> 仓库地址：<https://github.com/passant1/PyTorchDiagQA>

---

## 目录结构

```
├── 实验代码+报告/
│   ├── 实验一_中文分词/        # 前馈神经网络中文分词
│   ├── 实验二_句子相似度/      # TF-IDF 句向量与余弦相似度
│   └── 实验三_文本分类/        # TextCNN 文本分类
│
├── 期末报告/
│   ├── source/                 # PyTorchDiagQA 源码
│   └── 打开里面的exe执行/      # 打包好的可执行程序
│
└── README.md
```

---

## 实验一：中文分词

基于前馈神经网络（FFNN）实现中文分词，使用从 199801 大语料抽取的 1200 句标注数据。

- **模型**: 前馈神经网络（FFNN）
- **数据**: `data/segmentation_corpus.txt`

```bash
cd 实验代码+报告/实验一_中文分词
pip install -r requirements.txt
python run_exp1.py
```

详见 [实验一 README](实验代码+报告/实验一_中文分词/README.md)

---

## 实验二：句子相似度

基于 TF-IDF 句向量和余弦相似度计算句子间的语义相似度。

- **方法**: TF-IDF + 余弦相似度
- **数据**: 6 句真实语料句子

```bash
cd 实验代码+报告/实验二_句子相似度
pip install -r requirements.txt
python run_exp2.py
```

详见 [实验二 README](实验代码+报告/实验二_句子相似度/README.md)

---

## 实验三：文本分类

基于 TextCNN 实现文本分类（体育、科技、教育三类），附带训练损失曲线和混淆矩阵。

- **模型**: TextCNN
- **数据**: 180 条弱标注数据（体育 / 科技 / 教育）

```bash
cd 实验代码+报告/实验三_文本分类
pip install -r requirements.txt
python run_exp3.py
```

详见 [实验三 README](实验代码+报告/实验三_文本分类/README.md)

---

## 期末大作业：PyTorch 智能报错诊断问答系统（PyTorchDiagQA）

基于 PyTorch 官方文档知识图谱的深度学习报错诊断问答系统，融合 **RAG 检索增强生成**、**知识图谱**、**FAQ 检索**和**大模型润色**，提供桌面 GUI（Tkinter）和命令行交互。

### 功能

| 功能 | 说明 |
|------|------|
| 🔍 RAG 检索增强问答 | 从知识图谱 + FAQ 证据块检索上下文，生成诊断答案 |
| 🧠 知识图谱 | 40+ 节点、90+ 条边，覆盖 PyTorch 十大高频错误 |
| 📚 FAQ 检索 | TF-IDF + 余弦相似度匹配 32 条预置问答 |
| 🤖 大模型润色 | 支持 DeepSeek / OpenAI 兼容 API |
| 🖥️ 桌面 GUI | Tkinter 图形界面，支持离线运行 |
| 🔊 语音播报 | pyttsx3 离线朗读答案 |

### 覆盖的问题类型

1. CUDA 不可用（`torch.cuda.is_available()` 返回 False）
2. CUDA 显存不足（`CUDA out of memory`）
3. PyTorch 安装版本问题
4. DataLoader 参数问题
5. DataLoader 多进程问题（Windows `BrokenPipeError`）
6. 模型保存（`torch.save()`）
7. 模型加载（`torch.load()` / `map_location`）
8. `state_dict` 不匹配（Missing / Unexpected keys）
9. `torch.load` 安全问题（`weights_only`）
10. `CrossEntropyLoss` 输入问题

### 快速开始

```bash
cd 期末报告/source

# 安装依赖
pip install -r requirements.txt
pip install jieba scikit-learn numpy pyttsx3

# 启动桌面 GUI
python main.py --gui

# 或命令行交互
python main.py --cli

# 单次问答
python main.py --ask "torch.cuda.is_available 返回 False 怎么办？"

# 启动 FastAPI 后端
python main.py --server
```

### 配置大模型 API

编辑 `config.yaml` 或设置环境变量：

```bash
export API_KEY=your-api-key
export BASE_URL=https://api.deepseek.com
export MODEL_NAME=deepseek-v4-flash
```

不配置 API Key 时系统会自动运行在**离线模式**，使用内置知识图谱和 FAQ 回答问题。

详见 [PyTorchDiagQA README](期末报告/source/README.md)

---

## 环境要求

- Python 3.8+
- 依赖包：`jieba`, `scikit-learn`, `numpy`, `torch`, `pyyaml`, `pyttsx3`
- 详见各子目录的 `requirements.txt`

---

## 许可证

本仓库仅用于课程作业提交，保留所有权利。
