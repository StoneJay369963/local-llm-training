"""
多格式模型加载器
支持HuggingFace (safetensors/pyTorch)、GGUF、GPTQ、AWQ等多种格式
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import torch

logger = logging.getLogger(__name__)


class ModelFormat(str, Enum):
    """支持的模型格式"""
    SAFETENSORS = "safetensors"      # HuggingFace安全格式
    PYTORCH = "pytorch"             # PyTorch格式 (.pt, .bin)
    GGUF = "gguf"                   # llama.cpp量化格式
    GPTQ = "gptq"                   # GPTQ量化格式
    AWQ = "awq"                     # AWQ量化格式
    AUTO = "auto"                   # 自动检测


class ModelLoader:
    """
    多格式模型加载器
    自动检测模型格式并选择最佳加载方式
    """

    # 格式检测规则
    FORMAT_INDICATORS = {
        ".safetensors": ModelFormat.SAFETENSORS,
        ".bin": ModelFormat.PYTORCH,
        ".pt": ModelFormat.PYTORCH,
        ".gguf": ModelFormat.GGUF,
        ".ggml": ModelFormat.GGUF,  # 旧格式
        "-4bit.pt": ModelFormat.GPTQ,
        "-8bit.pt": ModelFormat.GPTQ,
        ".awq": ModelFormat.AWQ,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loaders = {
            ModelFormat.SAFETENSORS: self._load_safetensors,
            ModelFormat.PYTORCH: self._load_pytorch,
            ModelFormat.GGUF: self._load_gguf,
            ModelFormat.GPTQ: self._load_gptq,
            ModelFormat.AWQ: self._load_awq,
        }

    def detect_format(self, model_path: Union[str, Path]) -> ModelFormat:
        """
        自动检测模型格式

        Args:
            model_path: 模型路径

        Returns:
            检测到的格式
        """
        model_path = Path(model_path)

        # 首先检查文件扩展名
        for ext, fmt in self.FORMAT_INDICATORS.items():
            if str(model_path).endswith(ext):
                logger.info(f"通过扩展名检测到格式: {fmt}")
                return fmt

        # 检查目录中的文件
        if model_path.is_dir():
            files = list(model_path.glob("*"))

            # 检查safetensors
            if any(f.suffix == ".safetensors" for f in files):
                return ModelFormat.SAFETENSORS

            # 检查GGUF
            if any(f.suffix == ".gguf" for f in files):
                return ModelFormat.GGUF

            # 检查GPTQ
            if any("-4bit" in f.name or "-8bit" in f.name for f in files):
                return ModelFormat.GPTQ

            # 检查AWQ
            if any(f.suffix == ".awq" for f in files):
                return ModelFormat.AWQ

            # 检查config.json（通常是HuggingFace格式）
            if (model_path / "config.json").exists():
                return ModelFormat.SAFETENSORS

            # 默认为PyTorch
            return ModelFormat.PYTORCH

        # 检查单个文件
        if model_path.suffix == ".gguf":
            return ModelFormat.GGUF

        return ModelFormat.PYTORCH

    def load(
        self,
        model_path: Union[str, Path],
        format_hint: Optional[ModelFormat] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载模型

        Args:
            model_path: 模型路径
            format_hint: 格式提示（可选）
            **kwargs: 传递给加载器的额外参数

        Returns:
            (model, tokenizer)
        """
        model_path = Path(model_path)

        # 确定格式
        if format_hint and format_hint != ModelFormat.AUTO:
            fmt = format_hint
        else:
            fmt = self.detect_format(model_path)

        logger.info(f"使用格式加载器: {fmt} for {model_path}")

        # 调用对应的加载器
        if fmt in self.loaders:
            return self.loaders[fmt](model_path, **kwargs)
        else:
            raise ValueError(f"不支持的格式: {fmt}")

    def _load_safetensors(
        self,
        model_path: Path,
        **kwargs
    ) -> Tuple[Any, Any]:
        """加载HuggingFace safetensors格式"""
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoConfig,
        )

        device = kwargs.get("device", "auto")
        torch_dtype = kwargs.get("torch_dtype", torch.float16)
        trust_remote_code = kwargs.get("trust_remote_code", False)

        logger.info(f"加载safetensors模型: {model_path}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        return model, tokenizer

    def _load_pytorch(
        self,
        model_path: Path,
        **kwargs
    ) -> Tuple[Any, Any]:
        """加载PyTorch格式"""
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        device = kwargs.get("device", "auto")
        torch_dtype = kwargs.get("torch_dtype", torch.float32)

        logger.info(f"加载PyTorch模型: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device,
            torch_dtype=torch_dtype,
        )

        return model, tokenizer

    def _load_gguf(
        self,
        model_path: Path,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载GGUF格式 (llama.cpp)
        需要安装: pip install llama-cpp-python
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("请安装llama-cpp-python: pip install llama-cpp-python")
            raise ImportError(
                "GGUF格式需要安装llama-cpp-python\n"
                "运行: pip install llama-cpp-python\n"
                "Windows用户可能需要安装GPU版本: pip install llama-cpp-python --force-reinstall --no-cache-dir"
            )

        device = kwargs.get("device", "cpu")
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1表示全部GPU
        n_threads = kwargs.get("n_threads", None)
        verbose = kwargs.get("verbose", False)

        logger.info(f"加载GGUF模型: {model_path}")

        # 确定GPU加速
        if device == "cuda" and torch.cuda.is_available():
            # Windows使用CUDA
            try:
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    verbose=verbose,
                )
            except Exception as e:
                logger.warning(f"CUDA加载失败，回退到CPU: {e}")
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=verbose,
                )
        elif device == "mps" and torch.backends.mps.is_available():
            # macOS使用Metal
            llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,  # Metal支持
                n_threads=n_threads,
                verbose=verbose,
            )
        else:
            # CPU
            llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=verbose,
            )

        # 创建兼容接口
        model = GGUFModelWrapper(llm)
        tokenizer = GGUFTokenizerWrapper(llm)

        return model, tokenizer

    def _load_gptq(
        self,
        model_path: Path,
        **kwargs
    ) -> Tuple[Any, Any]:
        """加载GPTQ量化格式"""
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("请安装auto-gptq: pip install auto-gptq")
            raise ImportError(
                "GPTQ格式需要安装auto-gptq\n"
                "运行: pip install auto-gptq"
            )

        device = kwargs.get("device", "cuda:0")
        use_triton = kwargs.get("use_triton", False)

        logger.info(f"加载GPTQ模型: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        model = AutoGPTQForCausalLM.from_quantized(
            str(model_path),
            device=device,
            use_triton=use_triton,
        )

        return model, tokenizer

    def _load_awq(
        self,
        model_path: Path,
        **kwargs
    ) -> Tuple[Any, Any]:
        """加载AWQ量化格式"""
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("请安装awq: pip install autoawq")
            raise ImportError(
                "AWQ格式需要安装autoawq\n"
                "运行: pip install autoawq"
            )

        device = kwargs.get("device", "cuda")

        logger.info(f"加载AWQ模型: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        model = AutoAWQForCausalLM.from_pretrained(
            str(model_path),
            device_map=device,
        )

        return model, tokenizer


# ============ GGUF模型包装器 ============

class GGUFModelWrapper:
    """
    GGUF模型包装器
    提供与HuggingFace模型兼容的接口
    """

    def __init__(self, llm):
        self.llm = llm
        self.config = llm.model_kwargs if hasattr(llm, 'model_kwargs') else {}
        self.device = "cpu"  # llama.cpp在内部管理设备

    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        **kwargs
    ):
        """生成文本"""
        # 将input_ids转换为文本
        from transformers import PreTrainedTokenizer

        if isinstance(input_ids, torch.Tensor):
            # llama.cpp直接接受文本
            # 这里需要从input_ids解码
            pass

        # 对于GGUF，直接使用文本输入
        raise NotImplementedError(
            "GGUF模型请使用 generate_text() 方法"
        )

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
        **kwargs
    ) -> str:
        """生成文本（直接文本输入）"""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
        )

        return output["choices"][0]["text"]

    def __call__(self, **kwargs):
        """兼容HuggingFace模型的调用方式"""
        if "input_ids" in kwargs:
            prompt = kwargs.get("prompt", "")
            max_tokens = kwargs.get("max_new_tokens", 100)
            temperature = kwargs.get("temperature", 0.7)

            text = self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return type('obj', (object,), {'sequences': [[text]]})()

        return None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def eval(self):
        return self

    def train(self):
        return self


class GGUFTokenizerWrapper:
    """
    GGUF分词器包装器
    提供与HuggingFace分词器兼容的接口
    """

    def __init__(self, llm):
        self.llm = llm

        # 尝试获取vocab信息
        if hasattr(llm, 'model'):
            self.n_vocab = llm.model.n_tokens()
        else:
            self.n_vocab = 32000

    def encode(self, text: str, **kwargs) -> list:
        """编码文本"""
        # 简单的字符级编码（实际应使用模型的vocab）
        # llama.cpp内部处理分词
        return [0] * len(text)

    def decode(self, token_ids: list, **kwargs) -> str:
        """解码token"""
        return "".join(chr(min(t, 255)) for t in token_ids if t > 0)

    def __call__(self, text: str, **kwargs):
        """兼容HuggingFace分词器"""
        return {
            "input_ids": self.encode(text),
            "attention_mask": [1] * len(text),
        }

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def bos_token_id(self) -> int:
        return 1


# ============ 模型格式信息 ============

MODEL_FORMAT_INFO = {
    ModelFormat.SAFETENSORS: {
        "name": "SafeTensors",
        "description": "HuggingFace安全张量格式，加载快且安全",
        "extension": [".safetensors"],
        "loader": "transformers",
        "quantized": False,
        "pros": ["加载快速", "内存安全", "HuggingFace原生支持"],
        "cons": ["模型较大", "需要完整显存"],
    },
    ModelFormat.PYTORCH: {
        "name": "PyTorch",
        "description": "标准PyTorch模型格式",
        "extension": [".bin", ".pt"],
        "loader": "transformers",
        "quantized": False,
        "pros": ["兼容性最好", "支持全部功能"],
        "cons": ["文件较大", "加载较慢"],
    },
    ModelFormat.GGUF: {
        "name": "GGUF",
        "description": "llama.cpp量化格式，CPU/GPU高效推理",
        "extension": [".gguf"],
        "loader": "llama-cpp-python",
        "quantized": True,
        "pros": ["支持量化", "CPU高效", "显存要求低", "部署简单"],
        "cons": ["训练不支持", "精度略低"],
    },
    ModelFormat.GPTQ: {
        "name": "GPTQ",
        "description": "GPTQ量化格式，4/8bit量化",
        "extension": ["-4bit.pt", "-8bit.pt"],
        "loader": "auto-gptq",
        "quantized": True,
        "pros": ["4/8bit量化", "显存占用低", "推理速度快"],
        "cons": ["需要校准数据", "可能需要triton"],
    },
    ModelFormat.AWQ: {
        "name": "AWQ",
        "description": "AWQ激活感知量化",
        "extension": [".awq"],
        "loader": "autoawq",
        "quantized": True,
        "pros": ["精度较高", "显存占用低"],
        "cons": ["相对较新", "支持有限"],
    },
}


def print_format_info():
    """打印支持的格式信息"""
    print("\n" + "=" * 60)
    print("支持的模型格式")
    print("=" * 60)

    for fmt, info in MODEL_FORMAT_INFO.items():
        print(f"\n【{info['name']}】({fmt.value})")
        print(f"  描述: {info['description']}")
        print(f"  扩展名: {', '.join(info['extension'])}")
        print(f"  量化: {'是' if info['quantized'] else '否'}")
        print(f"  优点: {', '.join(info['pros'])}")
        print(f"  缺点: {', '.join(info['cons'])}")

    print("\n" + "=" * 60)
