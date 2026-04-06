"""
配置管理系统核心模块
使用Pydantic进行配置验证和类型安全
"""

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
from enum import Enum
import yaml
import json
import os


class ModelType(str, Enum):
    """支持的模型类型"""
    TRANSFORMER = "transformer"
    GPT2 = "gpt2"
    LLAMA = "llama"
    BLOOM = "bloom"
    CUSTOM = "custom"


class TrainingMode(str, Enum):
    """训练模式"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    LORA = "lora"
    QLORA = "qlora"


class DeviceType(str, Enum):
    """设备类型"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class QuantizationType(str, Enum):
    """量化类型"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


class DataFormat(str, Enum):
    """数据格式"""
    CHATML = "chatml"
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    RAW = "raw"


# ============ 基础配置 ============

class PathConfig(BaseModel):
    """路径配置"""
    project_root: Path = Field(default_factory=Path, description="项目根目录")
    data_dir: Path = Field(default_factory=lambda: Path("data"), description="数据目录")
    model_dir: Path = Field(default_factory=lambda: Path("models"), description="模型目录")
    log_dir: Path = Field(default_factory=lambda: Path("logs"), description="日志目录")
    cache_dir: Path = Field(default_factory=lambda: Path("cache"), description="缓存目录")
    output_dir: Path = Field(default_factory=lambda: Path("output"), description="输出目录")

    @field_validator('data_dir', 'model_dir', 'log_dir', 'cache_dir', 'output_dir', mode='before')
    @classmethod
    def resolve_path(cls, v):
        """确保路径相对于项目根目录"""
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_dirs(self):
        """创建所有必要的目录"""
        for path in [self.data_dir, self.model_dir, self.log_dir, self.cache_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


class HardwareConfig(BaseModel):
    """硬件配置"""
    device: DeviceType = DeviceType.CUDA
    num_workers: int = Field(default=4, ge=1, le=32, description="数据加载线程数")
    pin_memory: bool = Field(default=True, description="固定内存加速")
    persistent_workers: bool = Field(default=True, description="持久化数据加载进程")
    mixed_precision: bool = Field(default=True, description="混合精度训练")
    gradient_checkpointing: bool = Field(default=True, description="梯度检查点")
    max_memory: Optional[Dict[str, str]] = Field(default=None, description="显存限制")

    class Config:
        use_enum_values = True


# ============ 模型配置 ============

class ModelArchitectureConfig(BaseModel):
    """模型架构配置"""
    model_type: ModelType = ModelType.TRANSFORMER
    vocab_size: int = Field(default=32000, ge=1000, description="词表大小")
    hidden_size: int = Field(default=512, ge=64, description="隐藏层维度")
    num_hidden_layers: int = Field(default=8, ge=1, description="Transformer层数")
    num_attention_heads: int = Field(default=8, ge=1, description="注意力头数")
    intermediate_size: int = Field(default=2048, ge=64, description="FFN中间层维度")
    max_position_embeddings: int = Field(default=2048, ge=128, description="最大位置编码长度")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout比例")
    activation: str = Field(default="gelu", description="激活函数")
    layer_norm_eps: float = Field(default=1e-6, description="LayerNorm epsilon")

    class Config:
        use_enum_values = True

    def calculate_parameters(self) -> int:
        """计算模型参数量（估算）"""
        # 简化的参数量估算
        embedding_params = self.vocab_size * self.hidden_size * 2  # embedding + pos
        attention_params = (
            4 * self.hidden_size * self.hidden_size * self.num_hidden_layers  # QKV + O
        )
        ffn_params = (
            2 * self.intermediate_size * self.hidden_size * self.num_hidden_layers
        )
        layer_norm_params = 4 * self.hidden_size * self.num_hidden_layers
        return embedding_params + attention_params + ffn_params + layer_norm_params


class PretrainModelConfig(ModelArchitectureConfig):
    """预训练模型配置"""
    initializer_range: float = Field(default=0.02, description="初始化范围")
    tie_word_embeddings: bool = Field(default=False, description="绑定词嵌入")
    use_cache: bool = Field(default=False, description="使用KV缓存")


class FinetuneModelConfig(BaseModel):
    """微调模型配置"""
    name_or_path: str = Field(default="", description="模型名称或路径")
    use_fast: bool = Field(default=True, description="使用快速分词器")
    trust_remote_code: bool = Field(default=False, description="信任远程代码")
    use_cache: bool = Field(default=False, description="使用KV缓存")
    revision: Optional[str] = Field(default=None, description="模型版本")
    cache_dir: Optional[Path] = Field(default=None, description="缓存目录")


class LoRAConfig(BaseModel):
    """LoRA配置"""
    enable: bool = Field(default=True, description="启用LoRA")
    r: int = Field(default=16, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="目标模块"
    )
    bias: str = Field(default="none", description="偏置处理")
    task_type: str = Field(default="CAUSAL_LM", description="任务类型")

    @field_validator('target_modules')
    @classmethod
    def validate_target_modules(cls, v):
        """验证目标模块"""
        valid_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head", "embed_tokens"
        ]
        for module in v:
            if module not in valid_modules:
                raise ValueError(f"无效的目标模块: {module}")
        return v


class QuantizationConfig(BaseModel):
    """量化配置"""
    enable: bool = Field(default=False, description="启用量化")
    load_in_4bit: bool = Field(default=False, description="4bit加载")
    load_in_8bit: bool = Field(default=False, description="8bit加载")
    bnb_4bit_compute_dtype: str = Field(default="bfloat16", description="计算精度")
    bnb_4bit_quant_type: str = Field(default="nf4", description="量化类型")
    bnb_4bit_use_double_quant: bool = Field(default=True, description="双重量化")


# ============ 训练配置 ============

class OptimizerConfig(BaseModel):
    """优化器配置"""
    optim: str = Field(default="adamw_torch", description="优化器类型")
    learning_rate: float = Field(default=1e-4, ge=1e-7, le=1.0, description="学习率")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="权重衰减")
    adam_beta1: float = Field(default=0.9, ge=0.0, le=1.0, description="Adam beta1")
    adam_beta2: float = Field(default=0.999, ge=0.0, le=1.0, description="Adam beta2")
    adam_epsilon: float = Field(default=1e-8, ge=1e-10, description="Adam epsilon")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="梯度裁剪")
    lr_scheduler_type: str = Field(default="cosine", description="学习率调度器")
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="预热比例")
    warmup_steps: int = Field(default=100, ge=0, description="预热步数")


class TrainingConfig(BaseModel):
    """训练配置"""
    mode: TrainingMode = TrainingMode.PRETRAIN
    output_dir: Path = Field(default_factory=lambda: Path("output"), description="输出目录")
    num_train_epochs: int = Field(default=3, ge=1, description="训练轮数")
    per_device_train_batch_size: int = Field(default=4, ge=1, description="训练batch size")
    per_device_eval_batch_size: int = Field(default=4, ge=1, description="评估batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="梯度累积步数")
    eval_accumulation_steps: Optional[int] = Field(default=None, description="评估梯度累积")
    logging_steps: int = Field(default=10, ge=1, description="日志记录步数")
    save_steps: int = Field(default=500, ge=1, description="保存步数")
    eval_steps: int = Field(default=500, ge=1, description="评估步数")
    save_total_limit: int = Field(default=3, ge=1, description="最多保存的checkpoint数")
    fp16: bool = Field(default=False, description="FP16混合精度")
    bf16: bool = Field(default=False, description="BF16混合精度")
    seed: int = Field(default=42, description="随机种子")
    load_best_model_at_end: bool = Field(default=True, description="训练结束时加载最佳模型")
    metric_for_best_model: str = Field(default="eval_loss", description="最佳模型指标")
    greater_is_better: bool = Field(default=False, description="指标是否越大越好")
    group_by_length: bool = Field(default=True, description="按长度分组")
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"], description="日志报告目标")

    class Config:
        use_enum_values = True


# ============ 数据配置 ============

class DatasetConfig(BaseModel):
    """数据集配置"""
    train_path: Optional[Path] = Field(default=None, description="训练数据路径")
    eval_path: Optional[Path] = Field(default=None, description="验证数据路径")
    test_path: Optional[Path] = Field(default=None, description="测试数据路径")
    data_format: DataFormat = DataFormat.RAW
    max_samples: Optional[int] = Field(default=None, description="最大样本数")
    streaming: bool = Field(default=False, description="流式加载")
    shuffle_buffer_size: int = Field(default=10000, ge=100, description="缓冲区大小")
    overwrite_cache: bool = Field(default=False, description="覆盖缓存")

    class Config:
        use_enum_values = True


class DataProcessingConfig(BaseModel):
    """数据处理配置"""
    min_text_length: int = Field(default=100, ge=1, description="最小文本长度")
    max_text_length: int = Field(default=50000, ge=1, description="最大文本长度")
    remove_html: bool = Field(default=True, description="移除HTML")
    remove_urls: bool = Field(default=True, description="移除URL")
    remove_emails: bool = Field(default=True, description="移除邮箱")
    normalize_whitespace: bool = Field(default=True, description="标准化空白")
    lowercase: bool = Field(default=False, description="转小写")
    remove_special_chars: bool = Field(default=False, description="移除特殊字符")
    min_sentence_count: int = Field(default=3, ge=1, description="最小句子数")


class TokenizerConfig(BaseModel):
    """分词器配置"""
    type: str = Field(default="bpe", description="分词器类型")
    vocab_file: Optional[Path] = Field(default=None, description="词表文件")
    max_length: int = Field(default=2048, ge=16, description="最大长度")
    padding: str = Field(default="max_length", description="填充方式")
    truncation: bool = Field(default=True, description="截断")
    padding_side: str = Field(default="right", description="填充方向")


# ============ 网络API配置 ============

class APIProvider(str, Enum):
    """API提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


class APIConfig(BaseModel):
    """API配置"""
    provider: APIProvider = APIProvider.OPENAI
    base_url: Optional[str] = Field(default=None, description="API基础URL")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称")
    timeout: int = Field(default=60, ge=1, description="超时时间(秒)")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    max_tokens: int = Field(default=2048, ge=1, description="最大生成长度")

    class Config:
        use_enum_values = True


class CloudTrainingConfig(BaseModel):
    """云端训练配置"""
    enable: bool = Field(default=False, description="启用云端训练")
    api: APIConfig = Field(default_factory=APIConfig)
    enable_data_augmentation: bool = Field(default=False, description="启用数据增强")
    enable_quality_filter: bool = Field(default=False, description="启用质量过滤")
    generate_synthetic_data: bool = Field(default=False, description="生成合成数据")
    synthetic_data_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="合成数据比例")


# ============ 推理配置 ============

class InferenceConfig(BaseModel):
    """推理配置"""
    checkpoint: Optional[Path] = Field(default=None, description="检查点路径")
    max_new_tokens: int = Field(default=512, ge=1, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    top_k: int = Field(default=50, ge=0, description="Top-k采样")
    repetition_penalty: float = Field(default=1.1, ge=1.0, description="重复惩罚")
    do_sample: bool = Field(default=True, description="是否采样")
    num_beams: int = Field(default=1, ge=1, description="beam数量")


# ============ 主配置类 ============

class AppConfig(BaseSettings):
    """应用主配置"""
    version: str = Field(default="1.0.0", description="版本号")
    name: str = Field(default="LocalLLMTraining", description="应用名称")
    environment: str = Field(default="development", description="环境")

    # 子配置
    paths: PathConfig = Field(default_factory=PathConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: Optional[PretrainModelConfig] = Field(default=None, description="预训练模型配置")
    finetune_model: Optional[FinetuneModelConfig] = Field(default=None, description="微调模型配置")
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    cloud_training: CloudTrainingConfig = Field(default_factory=CloudTrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    class Config:
        env_prefix = "LLM_"
        env_nested_delimiter = "__"

    def save(self, path: Path) -> None:
        """保存配置到文件"""
        config_dict = self.model_dump(exclude_none=True, exclude={'version', 'name', 'environment'})

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'AppConfig':
        """从文件加载配置"""
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

        return cls(**config_dict)

    def ensure_directories(self):
        """确保所有目录存在"""
        self.paths.ensure_dirs()
        if self.training.output_dir:
            self.training.output_dir.mkdir(parents=True, exist_ok=True)


# ============ 配置工厂 ============

class ConfigFactory:
    """配置工厂"""

    @staticmethod
    def create_pretrain_config(**kwargs) -> AppConfig:
        """创建预训练配置"""
        config = AppConfig()
        config.model = PretrainModelConfig(**kwargs.get('model', {}))
        config.training.mode = TrainingMode.PRETRAIN
        config.training.output_dir = config.paths.model_dir / "pretrain_output"
        return config

    @staticmethod
    def create_finetune_config(model_name: str, **kwargs) -> AppConfig:
        """创建微调配置"""
        config = AppConfig()
        config.finetune_model = FinetuneModelConfig(name_or_path=model_name)
        config.training.mode = TrainingMode.LORA
        config.training.output_dir = config.paths.model_dir / "finetune_output"
        return config

    @staticmethod
    def create_qlora_config(model_name: str, **kwargs) -> AppConfig:
        """创建QLoRA配置"""
        config = AppConfig()
        config.finetune_model = FinetuneModelConfig(name_or_path=model_name)
        config.training.mode = TrainingMode.QLORA
        config.quantization.enable = True
        config.quantization.load_in_4bit = True
        config.training.output_dir = config.paths.model_dir / "qlora_output"
        return config

    @staticmethod
    def load_existing_config(config_path: Path) -> AppConfig:
        """加载现有配置"""
        return AppConfig.load(config_path)
