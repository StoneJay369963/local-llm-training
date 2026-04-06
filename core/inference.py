"""
推理引擎模块
支持多种模型格式：safetensors、PyTorch、GGUF、GPTQ、AWQ
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from core.config import AppConfig, InferenceConfig
from core.model_loader import (
    ModelLoader,
    ModelFormat,
    GGUFModelWrapper,
    print_format_info,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    prompt: str
    metrics: Dict[str, Any]
    model_format: Optional[ModelFormat] = None


class InferenceEngine:
    """
    多格式推理引擎

    支持的格式:
    - HuggingFace (safetensors, pytorch)
    - GGUF (llama.cpp)
    - GPTQ
    - AWQ
    """

    def __init__(
        self,
        config: AppConfig,
        model_path: Optional[Union[str, Path]] = None,
        format_hint: Optional[ModelFormat] = None,
    ):
        self.config = config
        self.model_path = model_path or config.inference.checkpoint or config.finetune_model.name_or_path
        self.format_hint = format_hint
        self.model_loader = ModelLoader()

        self.model: Any = None
        self.tokenizer: Any = None
        self.is_gguf: bool = False
        self.device = self._get_device()

        self._load_model()

    def _get_device(self) -> torch.device:
        """获取推理设备"""
        device = self.config.hardware.device

        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        """加载模型"""
        if not self.model_path:
            raise ValueError("未指定模型路径")

        model_path = Path(self.model_path)

        # 检测格式
        fmt = self.format_hint or self.model_loader.detect_format(model_path)
        logger.info(f"检测到模型格式: {fmt}")
        logger.info(f"模型路径: {model_path}")

        # 根据格式加载
        if fmt == ModelFormat.GGUF:
            self._load_gguf_model(model_path)
        else:
            self._load_huggingface_model(model_path)

    def _load_huggingface_model(self, model_path: Path):
        """加载HuggingFace格式模型"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 检查是否有LoRA权重
        lora_path = model_path / "adapter_model.bin"
        has_lora = lora_path.exists() or (model_path / "adapter_config.json").exists()

        if has_lora:
            logger.info("检测到LoRA权重，加载PEFT模型...")
            from peft import PeftModel

            base_path = str(model_path.parent) if lora_path.exists() else str(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            logger.info("加载完整模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()
        self.is_gguf = False

        # 打印信息
        self._print_model_info()

    def _load_gguf_model(self, model_path: Path):
        """加载GGUF格式模型"""
        logger.info("加载GGUF模型 (llama.cpp)...")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "\n" + "=" * 60 + "\n"
                "GGUF格式需要安装llama-cpp-python\n\n"
                "安装命令:\n"
                "  CPU版本: pip install llama-cpp-python\n"
                "  GPU版本 (NVIDIA): pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                "  macOS GPU: pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                "=" * 60
            )

        # GGUF参数
        n_ctx = self.config.tokenizer.max_length
        n_gpu_layers = -1 if self.config.hardware.device == "cuda" else 0

        # 创建llama.cpp实例
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # 创建包装器
        self.model = GGUFModelWrapper(self.llm)
        self.tokenizer = None  # GGUF使用内置分词
        self.is_gguf = True

        logger.info(f"GGUF模型已加载: {model_path.name}")

    def _print_model_info(self):
        """打印模型信息"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"模型参数量: {total_params / 1e9:.4f}B")
        except Exception:
            pass

        logger.info(f"设备: {self.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p采样
            top_k: top-k采样
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            num_beams: beam数量
            system_prompt: 系统提示

        Returns:
            生成的文本
        """
        # 使用配置中的默认值
        inference_config = self.config.inference
        max_new_tokens = max_new_tokens or inference_config.max_new_tokens
        temperature = temperature if temperature is not None else inference_config.temperature
        top_p = top_p or inference_config.top_p
        top_k = top_k or inference_config.top_k
        repetition_penalty = repetition_penalty or inference_config.repetition_penalty
        do_sample = do_sample if do_sample is not None else inference_config.do_sample
        num_beams = num_beams or inference_config.num_beams

        # 构建输入
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # 根据模型格式选择生成方法
        if self.is_gguf:
            return self._generate_gguf(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
            )
        else:
            return self._generate_huggingface(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
            )

    def _generate_huggingface(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        num_beams: int,
    ) -> str:
        """HuggingFace模型生成"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除原始prompt
        if generated_text.startswith(prompt):
            result = generated_text[len(prompt):].strip()
        else:
            result = generated_text.strip()

        return result

    def _generate_gguf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
    ) -> str:
        """GGUF模型生成"""
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p采样

        Yields:
            生成的token/text
        """
        if self.is_gguf:
            # GGUF流式生成
            for output in self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            ):
                if "choices" in output and len(output["choices"]) > 0:
                    text = output["choices"][0].get("text", "")
                    if text:
                        yield text
        else:
            # HuggingFace流式生成
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text in streamer:
                yield text

            thread.join()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        批量生成

        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大生成长度
            temperature: 温度

        Returns:
            生成的文本列表
        """
        if self.is_gguf:
            results = []
            for prompt in prompts:
                result = self._generate_gguf(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                )
                results.append(result)
            return results
        else:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            results = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if generated_text.startswith(prompts[i]):
                    result = generated_text[len(prompts[i]):].strip()
                else:
                    result = generated_text.strip()
                results.append(result)

            return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        对话生成

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]

        Returns:
            助手的回复
        """
        prompt = self._format_chat_messages(messages)
        return self.generate(prompt, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话消息"""
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg["content"]

            if role == "system":
                formatted.append(f"<|system|>\n{content}")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}")

        formatted.append("<|assistant|>")
        return "\n".join(formatted)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        fmt = self.format_hint or self.model_loader.detect_format(self.model_path)

        info = {
            "model_path": str(self.model_path),
            "model_format": fmt.value,
            "is_gguf": self.is_gguf,
            "device": str(self.device),
        }

        if not self.is_gguf and self.model:
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                info["total_parameters"] = total_params
                info["parameters_billions"] = total_params / 1e9
                info["dtype"] = str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
            except Exception:
                pass

        return info


class ModelInfo:
    """模型信息工具类"""

    @staticmethod
    def get_supported_formats() -> Dict[str, Any]:
        """获取支持的格式信息"""
        from core.model_loader import MODEL_FORMAT_INFO

        return {
            fmt.value: info for fmt, info in MODEL_FORMAT_INFO.items()
        }

    @staticmethod
    def detect_model_format(model_path: Union[str, Path]) -> ModelFormat:
        """检测模型格式"""
        loader = ModelLoader()
        return loader.detect_format(model_path)

    @staticmethod
    def print_supported_formats():
        """打印支持的格式"""
        print_format_info()


# ============ 便捷函数 ============

def load_model(
    model_path: Union[str, Path],
    config: Optional[AppConfig] = None,
    format_hint: Optional[ModelFormat] = None,
    **kwargs
) -> InferenceEngine:
    """
    便捷函数：加载模型

    Args:
        model_path: 模型路径
        config: 配置对象
        format_hint: 格式提示
        **kwargs: 额外参数

    Returns:
        InferenceEngine实例
    """
    if config is None:
        config = AppConfig()

    return InferenceEngine(
        config=config,
        model_path=model_path,
        format_hint=format_hint,
        **kwargs
    )


class CloudInferenceEngine:
    """
    云端推理引擎
    使用云端API进行推理，支持OpenAI、Anthropic等
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        model_name: Optional[str] = None,
    ):
        self.config = config or AppConfig()
        self.model_name = model_name or self.config.cloud_training.api.model_name
        self.api_config = self.config.cloud_training.api

        # 初始化API客户端
        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        from core.api_client import CloudAPIClient, OpenAIClient, AnthropicClient

        if self.api_config.provider == "openai":
            self.client = OpenAIClient(self.api_config)
        elif self.api_config.provider == "anthropic":
            self.client = AnthropicClient(self.api_config)
        else:
            raise ValueError(f"不支持的API提供商: {self.api_config.provider}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        同步生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p采样
            system_prompt: 系统提示

        Returns:
            生成的文本
        """
        import asyncio

        max_tokens = max_tokens or self.config.inference.max_new_tokens
        temperature = temperature if temperature is not None else self.config.inference.temperature
        top_p = top_p or self.config.inference.top_p

        # 转换为异步调用
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.client.generate(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            return result
        finally:
            loop.close()

    async def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        异步生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成长度
            temperature: 温度
            top_p: top-p采样
            system_prompt: 系统提示

        Returns:
            生成的文本
        """
        max_tokens = max_tokens or self.config.inference.max_new_tokens
        temperature = temperature if temperature is not None else self.config.inference.temperature
        top_p = top_p or self.config.inference.top_p

        return await self.client.generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    async def batch_generate_async(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        异步批量生成

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成长度
            temperature: 温度
            system_prompt: 系统提示

        Returns:
            生成的文本列表
        """
        return await self.client.batch_generate(
            prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens or self.config.inference.max_new_tokens,
            temperature=temperature or self.config.inference.temperature,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        对话生成

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]

        Returns:
            助手的回复
        """
        # 将消息格式转换为单个prompt
        prompt = self._format_chat_messages(messages)
        return self.generate(prompt, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话消息"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        formatted.append("Assistant:")
        return "\n\n".join(formatted)


class HybridInferenceEngine:
    """
    混合推理引擎
    同时支持本地模型和云端API，根据任务自动选择或组合使用
    """

    def __init__(
        self,
        local_model_path: Optional[Union[str, Path]] = None,
        config: Optional[AppConfig] = None,
        fallback_to_cloud: bool = True,
    ):
        """
        初始化混合推理引擎

        Args:
            local_model_path: 本地模型路径
            config: 配置对象
            fallback_to_cloud: 当本地模型不可用时是否回退到云端
        """
        self.config = config or AppConfig()
        self.fallback_to_cloud = fallback_to_cloud

        self.local_engine: Optional[InferenceEngine] = None
        self.cloud_engine: Optional[CloudInferenceEngine] = None

        # 尝试初始化本地引擎
        if local_model_path:
            try:
                self.local_engine = InferenceEngine(
                    config=self.config,
                    model_path=local_model_path,
                )
                logger.info(f"本地模型已加载: {local_model_path}")
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")
                if fallback_to_cloud:
                    self._init_cloud_engine()
        else:
            if fallback_to_cloud:
                self._init_cloud_engine()

    def _init_cloud_engine(self):
        """初始化云端引擎"""
        try:
            self.cloud_engine = CloudInferenceEngine(config=self.config)
            logger.info("云端推理引擎已初始化")
        except Exception as e:
            logger.error(f"云端引擎初始化失败: {e}")

    def generate(
        self,
        prompt: str,
        use_local: bool = True,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            use_local: 是否优先使用本地模型
            **kwargs: 传递给推理引擎的参数

        Returns:
            生成的文本
        """
        # 优先使用本地模型
        if use_local and self.local_engine:
            try:
                return self.local_engine.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"本地推理失败: {e}")

        # 回退到云端
        if self.cloud_engine:
            return self.cloud_engine.generate(prompt, **kwargs)

        raise RuntimeError("无可用的推理引擎")

    def generate_ensemble(
        self,
        prompt: str,
        local_weight: float = 0.7,
        **kwargs
    ) -> Dict[str, str]:
        """
        集成生成 - 同时使用本地和云端模型

        Args:
            prompt: 输入提示
            local_weight: 本地模型权重
            **kwargs: 传递给推理引擎的参数

        Returns:
            {"local": "...", "cloud": "...", "combined": "..."}
        """
        results = {"local": None, "cloud": None, "combined": None}

        # 本地生成
        if self.local_engine:
            try:
                results["local"] = self.local_engine.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"本地推理失败: {e}")

        # 云端生成
        if self.cloud_engine:
            try:
                results["cloud"] = self.cloud_engine.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"云端推理失败: {e}")

        # 组合结果
        if results["local"] and results["cloud"]:
            # 使用本地结果作为主要输出
            results["combined"] = results["local"]
            logger.info("集成生成完成，使用本地模型输出")
        elif results["local"]:
            results["combined"] = results["local"]
        elif results["cloud"]:
            results["combined"] = results["cloud"]

        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        对话生成

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            助手的回复
        """
        if self.local_engine:
            return self.local_engine.chat(messages, **kwargs)
        elif self.cloud_engine:
            return self.cloud_engine.chat(messages, **kwargs)

        raise RuntimeError("无可用的推理引擎")

    def get_available_engines(self) -> List[str]:
        """获取可用的推理引擎"""
        engines = []
        if self.local_engine:
            engines.append("local")
        if self.cloud_engine:
            engines.append("cloud")
        return engines
