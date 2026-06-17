# PyTorch 智能报错诊断问答系统

基于 PyTorch 官方文档知识图谱的深度学习报错诊断问答系统，面向 PyTorch 初学者和深度学习开发者。

> 当前版本采用 **RAG 优先** 的问答链路：系统先检索 `data/doc_chunks.json` 中的 FAQ、官方来源和知识图谱证据，再让大模型基于证据生成答案；没有 API 时会使用本地证据模板离线回答。知识图谱保留为结构化证据层和可视化展示层。

## 功能列表

- ✅ **知识图谱构建**: 从 PyTorch 官方文档自动/手动构建错误诊断知识图谱
- ✅ **RAG 检索增强问答**: 将 FAQ 与知识图谱节点转换为 `data/doc_chunks.json` 证据块，优先用检索证据生成答案
- ✅ **FAQ 检索**: 基于 TF-IDF 和余弦相似度的中文问题匹配
- ✅ **图谱问答**: 支持按问题类型、API、关系等多维度查询知识图谱
- ✅ **大模型润色**: 使用 OpenAI 兼容接口对答案进行语言润色（可选）
- ✅ **意图识别**: 自动识别用户查询原因、解决方法、检查命令等意图
- ✅ **实体抽取**: 自动识别 PyTorch API、错误类型、参数名等实体
- ✅ **桌面 GUI**: 基于 Tkinter 的图形化界面，支持离线运行
- ✅ **语音播报**: 使用 pyttsx3 离线朗读答案
- ✅ **语音输入**: 可选扩展模块，支持语音提问
- ✅ **FastAPI 后端**: 提供 REST API 接口
- ✅ **可执行打包**: 使用 PyInstaller 打包为独立 exe

## 项目结构

```
PyTorchDiagQA/
├── main.py                 # 主入口 (GUI / CLI / Server)
├── qa_engine.py            # 统一问答引擎
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── README.md               # 本文件
├── report.md               # 课程报告
├── build_exe.bat           # 打包脚本
├── PyDiag.spec             # PyInstaller 配置
│
├── backend/
│   └── app.py              # FastAPI 后端
│
├── frontend/
│   └── desktop_app.py      # Tkinter 桌面 GUI
│
├── modules/
│   ├── faq_retriever.py    # FAQ 检索模块
│   ├── kgqa.py             # 知识图谱问答模块
│   ├── intent_parser.py    # 意图识别与实体抽取
│   ├── llm_client.py       # 大模型 API 调用
│   ├── tts.py              # 语音合成
│   └── asr.py              # 语音识别（扩展）
│
├── kg_builder/
│   ├── sources.py          # 数据源管理 + 离线文本
│   ├── crawler.py          # 网页抓取
│   ├── cleaner.py          # 文本清洗
│   ├── extractor_rule.py   # 规则抽取
│   ├── extractor_llm.py    # 大模型辅助抽取
│   ├── graph_builder.py    # 图谱构建 + 种子图谱
│   └── export_csv.py       # CSV 导出
│
├── data/
│   ├── pytorch_sources.json  # 文档 URL 配置
│   ├── faq.json              # FAQ 数据 (32条)
│   ├── pytorch_kg.json       # 知识图谱数据
│   ├── nodes.csv             # 节点 CSV
│   ├── edges.csv             # 边 CSV
│   └── cache/                # 网页缓存
│
├── tests/
│   ├── test_faq.py
│   ├── test_kgqa.py
│   └── test_engine.py
│
├── utils/
│   └── path_utils.py       # 路径工具
│
└── assets/
    └── app.ico             # 应用图标
```

## 环境安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 jieba 分词依赖

```bash
pip install jieba scikit-learn numpy
```

### 3. 语音播报（可选）

```bash
pip install pyttsx3
```

### 4. 语音输入（可选）

```bash
pip install SpeechRecognition
pip install pipwin
pipwin install pyaudio
```

## 运行方式

### 方式一：桌面 GUI（推荐）

```bash
python main.py
# 或
python main.py --gui
```

### 方式二：命令行交互

```bash
python main.py --cli
```

### 方式三：单次问答

```bash
python main.py --ask "torch.cuda.is_available 返回 False 怎么办？"
```

## 构建知识图谱

### 自动构建

```bash
python main.py --build
```

这会：
1. 加载 `data/pytorch_sources.json` 中的文档 URL
2. 尝试网络抓取（失败则使用内置离线文本）
3. 使用规则抽取 API、参数、命令、代码块
4. 合并内置种子图谱
5. 生成 `data/pytorch_kg.json`
6. 同时导出 `data/nodes.csv` 和 `data/edges.csv`

### 在内置 GUI 中构建

点击 `构建/更新知识图谱` 按钮。

> **注意**: 即使完全离线，系统也会使用内置的种子图谱（40+ 节点，90+ 条边），保证核心功能可用。

## 配置大模型 API

### 1. 编辑 `config.yaml`

```yaml
llm:
  api_key: "your-api-key-here"  # 填入你的 API Key
  base_url: "https://api.siliconflow.cn/v1"
  model_name: "deepseek-ai/DeepSeek-V3"
  temperature: 0.3
  max_tokens: 1024
  enable_llm: true  # 设置为 true 启用
```

### 2. 使用环境变量

```bash
set LLM_API_KEY=your-api-key-here
# 或在 PowerShell 中:
$env:LLM_API_KEY="your-api-key-here"
```

### 3. 支持的 API 提供商

任何兼容 OpenAI 接口的服务均可使用，例如：
- **硅基流动**: `https://api.siliconflow.cn/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`
- **OpenAI**: `https://api.openai.com/v1`
- 其他兼容服务

> 如果不配置 API Key，系统将运行在**离线模式**，仅使用 FAQ 检索和知识图谱回答问题。

## 启动后端

```bash
# 方式一: 命令行启动
python main.py --server

# 方式二: 直接运行
python -m backend.app
```

启动后访问：
- API 地址: `http://127.0.0.1:18888`
- Swagger 文档: `http://127.0.0.1:18888/docs`

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/ask` | 问答接口 |
| GET | `/stats` | 系统统计 |
| GET | `/graph` | 知识图谱数据 |
| POST | `/build_kg` | 构建知识图谱 |

### 问答示例

```bash
curl -X POST http://127.0.0.1:18888/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "torch.cuda.is_available 返回 False 怎么办？", "use_llm": false}'
```

## 启动桌面前端

```bash
python main.py
```

桌面应用**不依赖后端**，可以直接加载本地引擎运行。如果后端已启动，也可以通过网络 API 调用。

## 打包 EXE

### 使用打包脚本（推荐）

双击运行 `build_exe.bat`

### 手动打包

```bash
# 1. 安装 PyInstaller
pip install pyinstaller

# 2. 构建知识图谱（确保数据文件存在）
python main.py --build

# 3. 打包（文件夹模式）
pyinstaller -D -w -n PyTorchDiag PyDiag.spec

# 4. 复制数据文件
xcopy /E data\*.json dist\PyTorchDiag\data\
xcopy /E data\*.csv dist\PyTorchDiag\data\
copy config.yaml dist\PyTorchDiag\
```

打包后目录结构：
```
dist/PyTorchDiag/
├── PyTorchDiag.exe    # 主程序
├── config.yaml        # 配置文件
├── data/              # 数据文件
├── assets/            # 资源文件
└── ...                # Python 运行时
```

## 常见问题

**Q: 启动时报 "jieba" 导入错误？**
A: `pip install jieba`

**Q: 知识图谱为空怎么办？**
A: 运行 `python main.py --build` 构建知识图谱

**Q: GUI 无法显示中文？**
A: 确保系统已安装中文字体（如 "Microsoft YaHei"）

**Q: 语音播报不工作？**
A: 安装 pyttsx3: `pip install pyttsx3`

**Q: 离线模式能用吗？**
A: 可以！系统内置了完整的种子知识图谱和 FAQ 数据，完全离线运行。

**Q: 如何添加更多 PyTorch 文档源？**
A: 编辑 `data/pytorch_sources.json`，添加新的 URL，然后重新构建知识图谱。

**Q: 打包后 exe 找不到数据文件？**
A: 系统已处理 PyInstaller 路径问题，使用 `utils/path_utils.py` 自动适配开发环境和打包环境。

## 覆盖的问题类型

| 序号 | 问题类型 | 核心 API/概念 |
|------|----------|--------------|
| 1 | CUDA 不可用 | `torch.cuda.is_available()` |
| 2 | CUDA 显存不足 | `torch.cuda.empty_cache()` |
| 3 | PyTorch 安装版本问题 | `torch.version.cuda` |
| 4 | DataLoader 参数问题 | `DataLoader` |
| 5 | DataLoader 多进程问题 | `num_workers` |
| 6 | 模型保存问题 | `torch.save()` |
| 7 | 模型加载问题 | `torch.load()` |
| 8 | state_dict 不匹配 | `load_state_dict()` |
| 9 | torch.load 安全问题 | `weights_only` |
| 10 | CrossEntropyLoss 输入问题 | `CrossEntropyLoss` |
