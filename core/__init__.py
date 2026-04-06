"""
LocalLLMTraining 核心模块
"""

from core.config import (
    AppConfig,
    ConfigFactory,
    TrainingMode,
    ModelType,
    DataFormat,
    HardwareConfig,
    ModelArchitectureConfig,
    PretrainModelConfig,
    FinetuneModelConfig,
    LoRAConfig,
    QuantizationConfig,
    OptimizerConfig,
    TrainingConfig,
    DatasetConfig,
    TokenizerConfig,
    CloudTrainingConfig,
    APIConfig,
)

from core.trainer import (
    BaseTrainer,
    PretrainRunner,
    FinetuneRunner,
    DistributedTrainer,
)

from core.inference import (
    InferenceEngine,
    CloudInferenceEngine,
    HybridInferenceEngine,
    GenerationResult,
)

from core.api_client import (
    CloudAPIClient,
    DataAugmentationPipeline,
    GeneratedSample,
    APIError,
)

from core.model_loader import (
    ModelLoader,
    ModelFormat,
    MODEL_FORMAT_INFO,
    GGUFModelWrapper,
)

from core.evaluator import (
    ModelEvaluator,
)

from core.exporter import (
    ModelExporter,
)

__all__ = [
    # 配置
    "AppConfig",
    "ConfigFactory",
    "TrainingMode",
    "ModelType",
    "DataFormat",
    "HardwareConfig",
    "ModelArchitectureConfig",
    "PretrainModelConfig",
    "FinetuneModelConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "DatasetConfig",
    "TokenizerConfig",
    "CloudTrainingConfig",
    "APIConfig",
    # 训练
    "BaseTrainer",
    "PretrainRunner",
    "FinetuneRunner",
    "DistributedTrainer",
    # 推理
    "InferenceEngine",
    "CloudInferenceEngine",
    "HybridInferenceEngine",
    "GenerationResult",
    # API
    "CloudAPIClient",
    "DataAugmentationPipeline",
    "GeneratedSample",
    "APIError",
    # 模型加载
    "ModelLoader",
    "ModelFormat",
    "MODEL_FORMAT_INFO",
    "GGUFModelWrapper",
    # 评估
    "ModelEvaluator",
    # 导出
    "ModelExporter",
]
