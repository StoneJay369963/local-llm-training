---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 30450220145272ba4a455b516c96ef3b0be52b08a102fa14c2e4935ab7bdd2b877f3107c0221009487a1aad579ec9d5d51c1bbdb5db62c85319855b279db13642218cfd9927fb7
    ReservedCode2: 3045022023aa2b3c5f76fe4bc3b66c331e80365accb23df9691f085e61099a0cf13d94bf022100fea1161beff09e381425e94d9c62ac080fd4af582b1a58013d74a0b6c5a4c907
---

# LocalLLMTraining

一个现代化的本地LLM训练框架，支持从零预训练、LoRA/QLoRA微调，以及通过接入云端API增强训练数据。让每个人都能拥有专属的本地语言模型。

## 核心特性

- **交互式配置向导**：通过简单的命令行交互，快速配置训练参数
- **模块化架构**：配置管理、训练引擎、推理服务分离，易于扩展
- **云端API集成**：支持接入OpenAI/Anthropic等云端模型，用于数据增强和质量过滤
- **Web服务API**：提供完整的REST API，支持远程控制和监控
- **多模式训练**：支持预训练、LoRA微调、QLoRA微调
- **硬件自适应**：自动检测GPU，支持混合精度和梯度检查点优化
- **多格式支持**：支持Safetensors、GGUF、GPTQ、AWQ等多种模型格式

## 支持的模型格式

本框架支持多种主流模型格式，可以自动检测并选择最佳加载方式：

| 格式 | 说明 | 量化支持 | 适用场景 | 显存要求 |
|------|------|---------|---------|---------|
| **Safetensors** | HuggingFace原生格式，加载快速安全 | 否 | 训练、微调、推理 | 高 |
| **PyTorch** | 标准PyTorch格式(.bin/.pt) | 否 | 通用场景 | 高 |
| **GGUF** | llama.cpp量化格式，CPU高效 | 是 | 本地推理、低显存 | 低 |
| **GPTQ** | 4/8bit量化格式 | 是 | 高效推理 | 中 |
| **AWQ** | 激活感知量化 | 是 | 高效推理 | 中 |

### 格式详细介绍

**GGUF格式（推荐用于推理）**
- 由llama.cpp推出的高效量化格式
- 支持Q2_K、Q4_K、Q5_K、Q6_K、Q8_0等多种量化级别
- 文件大小通常只有FP16的1/3到1/5
- 原生支持CPU推理，也支持GPU加速
- 适合在你的RTX 5060（8GB显存）上运行7B-13B模型

**Safetensors格式（推荐用于训练）**
- HuggingFace推出的安全张量格式
- 加载速度快，内存安全
- 支持HuggingFace生态全部功能
- 适合LoRA/QLoRA微调

**GPTQ/AWQ格式（推荐用于高效推理）**
- 4bit或8bit量化，显著降低显存占用
- 推理速度快，精度损失小
- 需要特定工具进行量化转换

### 查看支持的格式

```bash
# 查看所有支持的格式
python -m cli.interface formats

# 检测模型格式
python -m cli.interface model-info /path/to/model
```

### 安装额外依赖

```bash
# GGUF格式支持（推荐）
pip install llama-cpp-python

# GPU加速（NVIDIA CUDA）
pip install llama-cpp-python --force-reinstall --no-cache-dir

# GPTQ格式支持
pip install auto-gptq

# AWQ格式支持
pip install autoawq
```

## 项目架构

```
local-llm-training/
├── cli/                    # 命令行交互界面
│   └── interface.py        # 交互式配置向导和CLI命令
├── core/                   # 核心模块
│   ├── __init__.py        # 模块导出
│   ├── config.py          # 配置管理系统（Pydantic）
│   ├── trainer.py          # 训练引擎
│   ├── inference.py        # 推理引擎（多格式支持）
│   ├── model_loader.py     # 多格式模型加载器
│   └── api_client.py       # 云端API客户端
├── api/                    # Web服务
│   └── server.py           # FastAPI服务器
├── config/                 # 配置文件模板
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── data/                   # 数据目录
├── models/                 # 模型输出目录
├── logs/                   # 日志目录
├── requirements.txt        # Python依赖
└── README.md
```

## 安装指南

### Windows 用户（推荐）

**方式一：使用安装脚本**

```powershell
# 以管理员身份打开 PowerShell
cd local-llm-training

# 运行安装脚本
powershell -ExecutionPolicy Bypass -File install.ps1
```

**方式二：手动安装**

```powershell
# 1. 首先安装 PyTorch (使用 CUDA 11.8 预编译版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 安装核心依赖
pip install transformers datasets tokenizers accelerate peft trl
pip install bitsandbytes
pip install fastapi uvicorn[standard] python-multipart jinja2 aiohttp
pip install tenacity numpy pandas pyarrow
pip install tqdm pyyaml loguru
pip install pydantic pydantic-settings
pip install click inquirer

# 3. 安装可选依赖 (推荐)
pip install tensorboard wandb sentencepiece tiktoken regex

# 4. 安装量化支持 (可选，如果不需要GGUF/GPTQ可以跳过)
pip install auto-gptq optimum
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Linux / Mac 用户

```bash
# 克隆项目
git clone https://github.com/your-repo/local-llm-training.git
cd local-llm-training

# 运行安装脚本
bash install.sh

# 或手动安装
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 常见安装问题

**Q: autoawq 安装失败？**
> AWQ 在 Windows 上安装较复杂。如果不需要 AWQ 量化支持，可以跳过此包。

**Q: llama-cpp-python 安装失败？**
> Windows 用户建议使用预编译 wheel：
> https://github.com/abetlen/llama-cpp-python/releases

**Q: bitsandbytes 安装失败？**
> 尝试安装特定版本：`pip install bitsandbytes-windows`

## 快速开始

### 方式一：交互式初始化（推荐）

```bash
# 克隆项目
git clone https://github.com/your-repo/local-llm-training.git
cd local-llm-training

# 运行交互式设置向导
python -m cli.interface init
```

设置向导会引导你完成：

1. **路径配置**：设置数据目录、模型目录等
2. **硬件配置**：检测GPU，推荐最佳batch size和精度设置
3. **训练模式**：选择预训练、LoRA或QLoRA微调
4. **模型配置**：设置模型架构或选择开源模型
5. **数据配置**：指定训练数据或使用示例数据集
6. **优化器配置**：学习率、batch size等
7. **云端训练**：可选启用云端API辅助训练

### 方式二：命令行使用

```bash
# 查看系统信息
python -m cli.interface info

# 训练模型
python -m cli.interface train config.yaml --mode finetune

# 运行推理
python -m cli.interface infer config.yaml "你好，请介绍一下自己"

# 启动Web服务
python -m cli.interface serve config.yaml --host 0.0.0.0 --port 8000
```

### 方式三：Python API

```python
from core.config import AppConfig, ConfigFactory
from core.trainer import FinetuneRunner
from core.inference import InferenceEngine

# 方式1：从配置文件加载
config = AppConfig.load("config.yaml")

# 方式2：使用工厂创建
config = ConfigFactory.create_finetune_config("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 创建训练器
runner = FinetuneRunner(config)

# 开始训练
runner.train()

# 保存配置
config.save("my_config.yaml")
```

## 核心模块详解

### 配置管理系统

项目使用Pydantic进行类型安全的配置管理：

```python
from core.config import AppConfig, TrainingMode, LoRAConfig

config = AppConfig()

# 预训练配置
config.model = ModelArchitectureConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=8,
)

# LoRA配置
config.lora = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

# 训练配置
config.training.mode = TrainingMode.LORA
config.training.num_train_epochs = 3
```

### 训练引擎

```python
from core.trainer import PretrainRunner, FinetuneRunner

# 预训练（从零开始训练小型模型）
pretrain_runner = PretrainRunner(config)
pretrain_runner.train()

# 微调（基于开源模型）
finetune_runner = FinetuneRunner(config)
finetune_runner.train()

# 异步训练（配合Web服务）
await finetune_runner.train_async(progress_callback=callback)
```

### 推理引擎

```python
from core.inference import InferenceEngine

engine = InferenceEngine(config)

# 单条生成
result = engine.generate("你好，请介绍一下自己")

# 流式生成
for token in engine.generate_stream("写一首诗"):
    print(token, end="", flush=True)

# 对话
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你的吗？"},
    {"role": "user", "content": "介绍一下你自己"},
]
response = engine.chat(messages)
```

### 云端API集成

这是实现"接入网络模型训练本地模型"愿景的核心功能：

```python
from core.api_client import CloudAPIClient, DataAugmentationPipeline
from core.config import CloudTrainingConfig, APIConfig

# 配置云端API
cloud_config = CloudTrainingConfig(
    enable=True,
    api=APIConfig(
        provider="openai",
        api_key="your-api-key",  # 或使用环境变量
        model_name="gpt-3.5-turbo"
    ),
    enable_data_augmentation=True,
    enable_quality_filter=True,
    generate_synthetic_data=True
)

# 创建API客户端
client = CloudAPIClient(cloud_config)

# 生成训练样本
samples = await client.generate_samples(
    topic="人工智能",
    count=100,
    style="informative"
)

# 数据增强
augmented = await client.augment_data(original_texts, augmentation_ratio=0.5)

# 质量过滤
filtered = await client.filter_quality(all_texts)

# 数据增强流水线
pipeline = DataAugmentationPipeline(client)
result = await pipeline.process_dataset(
    input_path="raw_data.txt",
    output_path="augmented_data.txt",
    augment_ratio=0.3,
    filter_threshold=7.0
)

# 生成合成数据集
result = await pipeline.generate_synthetic_dataset(
    topics=["科技", "教育", "医疗"],
    output_path="synthetic_data.txt",
    samples_per_topic=100
)
```

## Web API服务

启动Web服务后，可以通过REST API进行远程控制：

```bash
# 启动服务
python -m api.server --config config.yaml
```

### 主要API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/config` | GET/POST | 获取/更新配置 |
| `/api/train` | POST | 启动训练 |
| `/api/train/status` | GET | 获取训练状态 |
| `/api/train/stop` | POST | 停止训练 |
| `/api/generate` | POST | 文本生成 |
| `/api/data/augment` | POST | 数据增强 |
| `/api/data/synthetic` | POST | 生成合成数据 |
| `/api/data/upload` | POST | 上传数据文件 |
| `/api/models` | GET | 列出可用模型 |

### API使用示例

```bash
# 启动训练
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"mode": "finetune", "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

# 获取训练状态
curl http://localhost:8000/api/train/status

# 文本生成
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，请介绍一下自己", "max_new_tokens": 512}'

# 数据增强
curl -X POST http://localhost:8000/api/data/augment \
  -H "Content-Type: application/json" \
  -d '{"input_path": "data/train.txt", "output_path": "data/augmented.txt"}'
```

## 云端训练架构

```
┌─────────────────────────────────────────────────────────────┐
│                      本地训练环境                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  CLI/Web    │    │   训练引擎   │    │   推理引擎   │    │
│  │   界面      │    │  (Trainer)  │    │  (Inference)│    │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘    │
│         │                  │                               │
│         └──────────────────┼───────────────────────────────┤
│                            │                               │
│  ┌─────────────────────────┴─────────────────────────────┐ │
│  │                    配置管理系统                          │ │
│  │                   (AppConfig)                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ API调用
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      云端API服务                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   数据生成   │    │   质量过滤   │    │  数据增强    │    │
│  │ (Synthetic) │    │  (Quality)  │    │ (Augment)   │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                              │
│  支持: OpenAI GPT-3.5/GPT-4, Anthropic Claude, 自定义API     │
└─────────────────────────────────────────────────────────────┘
```

通过这种架构，你可以：

1. **数据生成**：使用强大的云端模型生成高质量的训练数据
2. **质量过滤**：利用云端模型评估和过滤本地数据
3. **数据增强**：通过云端模型改写和扩展现有数据
4. **协同训练**：云端生成、本地训练的混合训练模式

## 硬件要求与优化

### 最低要求

- **GPU**：NVIDIA GPU，6GB+ 显存
- **内存**：8GB+
- **存储**：10GB+

### RTX 5060 (8GB) 推荐配置

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  fp16: true
  bf16: false

lora:
  r: 8
  lora_alpha: 16

quantization:
  enable: true
  load_in_4bit: true
```

### 显存优化技术

- **混合精度训练**：FP16/BF16减少显存占用
- **梯度检查点**：以计算换显存
- **QLoRA量化**：4bit量化大幅降低显存需求
- **梯度累积**：用小batch模拟大batch效果

## 数据准备

### 支持的数据格式

**原始文本（每行一段）**：
```
这是第一段文本
这是第二段文本
这是第三段文本
```

**Alpaca格式（JSON）**：
```json
[
    {
        "instruction": "请介绍...",
        "input": "",
        "output": "..."
    }
]
```

**ChatML格式（对话）**：
```json
[
    {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你的吗？"}
        ]
    }
]
```

### 数据清洗

框架内置数据清洗功能，支持：

- HTML标签移除
- URL和邮箱移除
- 文本长度过滤
- 句子数量验证
- MinHash去重

## 常见问题

### Q: 显存不足怎么办？

1. 启用QLoRA量化：`quantization.enable = true`
2. 减小batch size：`per_device_train_batch_size = 1`
3. 减小max_length：`tokenizer.max_length = 1024`
4. 启用梯度检查点：`gradient_checkpointing = true`

### Q: 如何使用自定义模型？

```python
from core.config import FinetuneModelConfig

config.finetune_model = FinetuneModelConfig(
    name_or_path="your-model-path-or-name"
)
```

### Q: 如何启用云端API？

```python
from core.config import CloudTrainingConfig, APIConfig

config.cloud_training = CloudTrainingConfig(
    enable=True,
    api=APIConfig(
        provider="openai",
        api_key="sk-...",
        model_name="gpt-3.5-turbo"
    )
)
```

### Q: 训练中断后如何恢复？

框架自动保存checkpoint，使用以下命令恢复：

```bash
python -m cli.interface train config.yaml
```

训练器会自动从最后一个checkpoint恢复。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 参考资源

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
