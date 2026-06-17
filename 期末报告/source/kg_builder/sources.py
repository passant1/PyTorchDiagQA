"""
数据源管理模块
管理 PyTorch 文档 URL 和离线文本
"""
import json


def load_source_urls(json_path):
    """从 JSON 配置文件加载文档 URL"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# 内置离线文本（网络不可用时使用）
# ==========================================
OFFLINE_DOCS = {
    "pytorch_get_started": """
PyTorch 安装与 CUDA 检查

安装 PyTorch:
pip install torch torchvision torchaudio

安装 CUDA 版本:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

检查 PyTorch 是否正确安装:
python -c "import torch; print(torch.__version__)"

检查 CUDA 是否可用:
import torch
print(torch.cuda.is_available())  # 返回 True 或 False
print(torch.version.cuda)  # CUDA 版本
print(torch.cuda.get_device_name(0))  # GPU 名称

如果 torch.cuda.is_available() 返回 False:
1. 确认安装了 NVIDIA 显卡
2. 确认安装了 CUDA 驱动
3. 确认 PyTorch 安装的是 CUDA 版本（而非 CPU 版本）
4. 使用 nvidia-smi 命令检查驱动和 CUDA 版本

常见问题:
- CPU 版本 PyTorch: pip install torch 默认在某些平台安装 CPU 版本
- 必须使用 --index-url 指定 CUDA 版本
""",

    "pytorch_data": """
torch.utils.data — Dataset 与 DataLoader

DataLoader 是 PyTorch 中用于批量加载数据的核心类。
它支持多进程数据加载、数据打乱、自定义批处理等功能。

DataLoader 主要参数:
- dataset: 数据集对象
- batch_size: 每个批次的大小
- shuffle: 是否打乱数据
- num_workers: 数据加载的子进程数量
- pin_memory: 是否将数据固定在内存中（GPU 训练时推荐 True）
- drop_last: 是否丢弃最后不足 batch_size 的批次
- collate_fn: 自定义批处理函数

num_workers 说明:
- num_workers=0 表示在主进程中加载数据（默认）
- num_workers>0 表示使用子进程加载数据
- Windows 上多进程有额外限制，建议在 if __name__ == '__main__' 保护下使用
- 过多 num_workers 可能导致内存占用过高
- 推荐设置: num_workers = 4 * num_gpus

常见问题:
- BrokenPipeError: Windows 下 DataLoader 多进程问题
  解决方法: num_workers=0 或将 DataLoader 放在 if __name__ == '__main__' 中
- pin_memory=True 在非 GPU 训练时无意义

示例代码:
from torch.utils.data import DataLoader, TensorDataset
import torch

data = torch.randn(100, 3, 32, 32)
labels = torch.randint(0, 10, (100,))
dataset = TensorDataset(data, labels)

loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
""",

    "pytorch_save_load": """
PyTorch 模型保存与加载

保存模型有两种方式:

方式一: 保存整个模型
torch.save(model, 'model.pth')
model = torch.load('model.pth')

方式二: 只保存 state_dict（推荐）
torch.save(model.state_dict(), 'model_weights.pth')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

推荐使用 state_dict 方式:
- 更灵活，可以跨设备加载
- 文件更小
- 加载时可以指定 map_location

保存检查点:
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

加载检查点:
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

torch.load 参数:
- map_location: 将模型加载到指定设备
  - torch.load('model.pth', map_location=torch.device('cpu'))  加载到 CPU
  - torch.load('model.pth', map_location='cuda:0')  加载到 GPU 0
- weights_only=True: PyTorch 2.0+ 推荐参数，只加载权重，提高安全性

跨设备加载:
# GPU 训练的模型加载到 CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
# CPU 训练的模型加载到 GPU
model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))
""",

    "pytorch_serialization": """
torch.load 安全与 weights_only 参数

从 PyTorch 2.0 开始，torch.load 增加了 weights_only 参数:
- torch.load('model.pth', weights_only=True) 只反序列化张量数据
- 设置为 True 可以防止 pickle 反序列化攻击
- 加载他人提供的模型文件时强烈建议使用 weights_only=True

安全问题:
- torch.load 底层使用 Python pickle，可能执行恶意代码
- 不信任来源的 .pth 文件不要轻易 torch.load
- PyTorch 2.6+ 默认 weights_only=True

与 torch.save 的关系:
torch.save(obj, 'file.pth') 使用 pickle 序列化
torch.load('file.pth') 使用 pickle 反序列化

推荐做法:
1. 只从信任来源加载模型
2. 使用 weights_only=True
3. 如果必须加载完整模型，在沙箱环境中进行
""",

    "pytorch_cross_entropy": """
torch.nn.CrossEntropyLoss 输入维度

CrossEntropyLoss 结合了 LogSoftmax 和 NLLLoss:
- 输入 logits: (N, C) 其中 N 是 batch_size, C 是类别数
- 输入 target: (N,) 其中每个值是类别索引 (0 到 C-1)
- 也支持 target: (N, d1, d2, ...) 用于更高维度的预测

重要:
- target 必须是 LongTensor (torch.long)
- target 的值必须在 [0, C-1] 范围内
- logits 不需要经过 softmax（CrossEntropyLoss 内部会处理）
- 不要对 logits 手动做 softmax + log！

常见错误:
RuntimeError: expected scalar type Long but found Float
→ target 需要是 LongTensor，使用 target.long() 转换

RuntimeError: CUDA error: device-side assert triggered
→ target 值超出了类别范围 [0, C-1]

维度不匹配:
- logits shape: (batch_size, num_classes)
- target shape: (batch_size,)
- 如果用了 one-hot encoding 的 target，不要用 CrossEntropyLoss，改用直接计算

与 NLLLoss 的区别:
- NLLLoss 接收 log-softmax 的输出
- CrossEntropyLoss 接收原始 logits
- CrossEntropyLoss = LogSoftmax + NLLLoss

示例代码:
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

# 正确用法
logits = torch.randn(32, 10)  # (batch_size=32, num_classes=10)
target = torch.randint(0, 10, (32,))  # (32,) LongTensor
loss = loss_fn(logits, target)

# 错误用法: target 是 FloatTensor
# target = torch.randn(32)  # 这会报错！
""",

    "pytorch_cuda_memory": """
CUDA 显存管理

CUDA out of memory 错误是 PyTorch 训练中最常见的问题之一。
原因通常是:
1. batch_size 过大
2. 模型过大
3. 梯度累积导致的显存泄漏
4. 多个进程同时使用 GPU

解决方法:
1. 减小 batch_size
2. 使用梯度累积来模拟大 batch_size
3. 使用 torch.cuda.empty_cache() 清理缓存
4. 使用 torch.no_grad() 在推理时关闭梯度计算
5. 使用 model.half() 或 AMP (自动混合精度) 减小显存占用
6. 使用 gradient checkpointing (torch.utils.checkpoint)

检查显存使用:
- nvidia-smi 命令行工具
- torch.cuda.memory_allocated() 已分配显存
- torch.cuda.memory_reserved() 已缓存显存
- torch.cuda.memory_summary() 详细显存报告

显存泄漏排查:
- 检查 loss 是否 detach
- 检查是否有全局变量持有 tensor
- 检查 DataLoader 的 pin_memory

torch.cuda.empty_cache():
- 释放 PyTorch 缓存的显存到 CUDA
- 不会影响 tensor 的生命周期
- 在 OOM 后调用可能有帮助

示例:
import torch

# 检查显存
print(f"已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 清理缓存
torch.cuda.empty_cache()
""",
}
