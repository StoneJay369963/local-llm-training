"""
云端API客户端模块
支持接入各种网络模型API，用于数据增强、质量过滤和合成数据生成
这是实现"接入网络模型训练本地模型"愿景的核心模块
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any, Callable, Iterator
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import APIConfig, CloudTrainingConfig, AppConfig

logger = logging.getLogger(__name__)


class APIError(Exception):
    """API错误基类"""
    pass


class RateLimitError(APIError):
    """速率限制错误"""
    pass


class AuthenticationError(APIError):
    """认证错误"""
    pass


class DataQuality(str, Enum):
    """数据质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GeneratedSample:
    """生成的样本"""
    text: str
    quality: DataQuality
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "api"


@dataclass
class QualityReport:
    """质量报告"""
    overall_score: float
    relevance_score: float
    coherence_score: float
    toxicity_score: float
    issues: List[str] = field(default_factory=list)


class BaseAPIClient(ABC):
    """API客户端基类"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        pass

    @abstractmethod
    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        pass


class OpenAIClient(BaseAPIClient):
    """OpenAI API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status == 401:
                    raise AuthenticationError("API密钥无效")
                elif response.status == 429:
                    raise RateLimitError("请求频率超限，请稍后重试")
                elif response.status != 200:
                    text = await response.text()
                    raise APIError(f"API请求失败: {response.status} - {text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
                # 简单的速率控制
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        return results


class AnthropicClient(BaseAPIClient):
    """Anthropic API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API密钥未设置")

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status == 401:
                    raise AuthenticationError("API密钥无效")
                elif response.status == 429:
                    raise RateLimitError("请求频率超限")

                result = await response.json()
                return result["content"][0]["text"]

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        return results


class CustomAPIClient(BaseAPIClient):
    """自定义API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.base_url = config.base_url

        if not self.base_url:
            raise ValueError("自定义API需要设置base_url")

        self.api_key = config.api_key

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/generate",
                headers=self._get_headers(),
                json=payload
            ) as response:
                result = await response.json()
                return result.get("text", "")

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, system_prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        return results


class CloudAPIClient:
    """
    云端API客户端管理器
    这是连接云端模型和本地训练的核心类
    """

    def __init__(self, cloud_config: CloudTrainingConfig):
        self.config = cloud_config
        self.api_config = cloud_config.api
        self.client: Optional[BaseAPIClient] = None

        if cloud_config.enable:
            self._initialize_client()

    def _initialize_client(self):
        """初始化API客户端"""
        provider = self.api_config.provider.lower()

        if provider == "openai":
            self.client = OpenAIClient(self.api_config)
        elif provider == "anthropic":
            self.client = AnthropicClient(self.api_config)
        elif provider == "custom":
            self.client = CustomAPIClient(self.api_config)
        else:
            raise ValueError(f"不支持的API提供商: {provider}")

        logger.info(f"已初始化 {provider} API客户端")

    async def generate_samples(
        self,
        topic: str,
        count: int = 10,
        style: str = "informative"
    ) -> List[GeneratedSample]:
        """
        生成训练样本

        Args:
            topic: 主题
            count: 生成数量
            style: 风格
        """
        if not self.client:
            raise APIError("API客户端未初始化")

        system_prompt = self._get_system_prompt(style)
        user_prompt = f"请生成{count}条关于'{topic}'的高质量文本，每条至少100字，风格{style}。以JSON数组格式输出。"

        try:
            response = await self.client.generate(
                user_prompt,
                system_prompt=system_prompt
            )

            # 解析JSON响应
            samples = self._parse_json_response(response, topic)
            return samples

        except Exception as e:
            logger.error(f"生成样本失败: {e}")
            return []

    async def augment_data(
        self,
        texts: List[str],
        augmentation_ratio: float = 0.5
    ) -> List[str]:
        """
        数据增强

        Args:
            texts: 原始文本
            augmentation_ratio: 增强比例
        """
        if not self.client:
            raise APIError("API客户端未初始化")

        augmented = []
        count = max(1, int(len(texts) * augmentation_ratio))

        system_prompt = "你是一个专业的数据增强专家。请对提供的文本进行改写，保持相同语义但使用不同的表达方式。"

        for text in texts[:count]:
            try:
                prompt = f"请改写以下文本，保持相同语义:\n\n{text}"
                result = await self.client.generate(prompt, system_prompt)
                augmented.append(result)
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.error(f"增强失败: {e}")
                augmented.append(text)

        return augmented

    async def filter_quality(
        self,
        texts: List[str]
    ) -> List[tuple]:
        """
        质量过滤

        Returns:
            [(text, quality_score), ...]
        """
        if not self.client:
            raise APIError("API客户端未初始化")

        results = []
        system_prompt = """你是一个文本质量评估专家。请评估每条文本的质量，从以下维度打分(0-10):
1. 相关性 - 与主题的相关程度
2. 连贯性 - 文本的逻辑性和流畅度
3. 安全性 - 是否包含有害内容

请以JSON格式输出: [{"score": 8.5, "relevance": 9, "coherence": 8, "safety": 10}]"""

        for text in texts:
            try:
                prompt = f"评估以下文本:\n\n{text[:500]}"
                response = await self.client.generate(prompt, system_prompt)

                quality = self._parse_quality_response(response)
                results.append((text, quality))

            except Exception as e:
                logger.error(f"质量评估失败: {e}")
                results.append((text, 5.0))

        return results

    async def generate_dialogue(
        self,
        topic: str,
        count: int = 10,
        rounds: int = 3
    ) -> List[Dict]:
        """
        生成对话数据

        Args:
            topic: 对话主题
            count: 对话数量
            rounds: 对话轮数
        """
        if not self.client:
            raise APIError("API客户端未初始化")

        dialogues = []
        system_prompt = """你是一个AI助手，正在与用户进行自然对话。请生成多轮对话数据，
包含user和assistant的交互。以JSON数组格式输出，每轮对话包含role和content。"""

        for i in range(count):
            try:
                prompt = f"生成一段关于'{topic}'的{rounds}轮对话"
                response = await self.client.generate(prompt, system_prompt)

                dialogue = self._parse_dialogue_response(response)
                dialogues.append(dialogue)
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"生成对话失败: {e}")

        return dialogues

    def _get_system_prompt(self, style: str) -> str:
        """获取系统提示"""
        prompts = {
            "informative": "你是一个知识渊博的AI助手，请生成准确、详细、有教育意义的内容。",
            "casual": "你是一个友好的AI助手，请生成轻松、自然、对话式的文本。",
            "technical": "你是一个技术专家，请生成专业、准确、技术性的内容。",
            "creative": "你是一个创意作家，请生成有创意、有想象力、引人入胜的内容。"
        }
        return prompts.get(style, prompts["informative"])

    def _parse_json_response(self, response: str, topic: str) -> List[GeneratedSample]:
        """解析JSON响应"""
        try:
            # 尝试提取JSON数组
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            samples = []
            for item in data:
                if isinstance(item, str):
                    text = item
                else:
                    text = item.get("text", item.get("content", ""))

                sample = GeneratedSample(
                    text=text,
                    quality=DataQuality.HIGH if len(text) > 100 else DataQuality.MEDIUM,
                    metadata={"topic": topic}
                )
                samples.append(sample)

            return samples

        except json.JSONDecodeError:
            logger.warning("JSON解析失败，尝试文本解析")
            # 备用解析
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            return [
                GeneratedSample(
                    text=line,
                    quality=DataQuality.MEDIUM,
                    metadata={"topic": topic}
                )
                for line in lines if len(line) > 50
            ]

    def _parse_quality_response(self, response: str) -> float:
        """解析质量评估响应"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]

            data = json.loads(response)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("score", 5.0)
            elif isinstance(data, dict):
                return data.get("score", 5.0)

        except json.JSONDecodeError:
            pass

        # 备用解析
        import re
        match = re.search(r'\d+\.?\d*', response)
        if match:
            score = float(match.group())
            return min(10.0, max(0.0, score))

        return 5.0

    def _parse_dialogue_response(self, response: str) -> Dict:
        """解析对话响应"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]

            data = json.loads(response)

            # 转换为标准格式
            messages = []
            for item in data:
                role = item.get("role", "assistant").lower()
                if role not in ["user", "assistant"]:
                    role = "assistant"
                messages.append({
                    "role": role,
                    "content": item.get("content", "")
                })

            return {"messages": messages}

        except json.JSONDecodeError:
            return {"messages": []}


class DataAugmentationPipeline:
    """
    数据增强流水线
    结合云端API进行端到端的数据处理
    """

    def __init__(self, cloud_client: CloudAPIClient):
        self.cloud_client = cloud_client

    async def process_dataset(
        self,
        input_path: Path,
        output_path: Path,
        augment_ratio: float = 0.3,
        filter_threshold: float = 7.0
    ) -> Dict[str, Any]:
        """
        处理数据集

        Args:
            input_path: 输入文件
            output_path: 输出文件
            augment_ratio: 增强比例
            filter_threshold: 质量阈值
        """
        # 读取数据
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"读取 {len(texts)} 条数据")

        # 质量过滤
        if self.cloud_client.config.enable_quality_filter:
            logger.info("开始质量过滤...")
            quality_results = await self.cloud_client.filter_quality(texts)
            filtered = [(t, q) for t, q in quality_results if q >= filter_threshold]
            texts = [t for t, _ in filtered]
            logger.info(f"质量过滤后剩余 {len(texts)} 条")

        # 数据增强
        augmented = []
        if self.cloud_client.config.enable_data_augmentation:
            logger.info("开始数据增强...")
            augmented = await self.cloud_client.augment_data(texts, augment_ratio)
            logger.info(f"生成 {len(augmented)} 条增强数据")

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts + augmented:
                f.write(text + '\n')

        return {
            "original_count": len(texts),
            "augmented_count": len(augmented),
            "total_count": len(texts) + len(augmented),
            "output_path": str(output_path)
        }

    async def generate_synthetic_dataset(
        self,
        topics: List[str],
        output_path: Path,
        samples_per_topic: int = 100
    ) -> Dict[str, Any]:
        """
        生成合成数据集

        Args:
            topics: 主题列表
            output_path: 输出路径
            samples_per_topic: 每个主题的样本数
        """
        all_samples = []

        for topic in topics:
            logger.info(f"生成 '{topic}' 主题数据...")

            # 批量生成
            batch_count = (samples_per_topic + 9) // 10
            for i in range(batch_count):
                count = min(10, samples_per_topic - i * 10)
                samples = await self.cloud_client.generate_samples(
                    topic=topic,
                    count=count,
                    style="informative"
                )
                all_samples.extend(samples)

                await asyncio.sleep(1)  # 速率控制

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(sample.text + '\n')

        return {
            "topic_count": len(topics),
            "total_samples": len(all_samples),
            "output_path": str(output_path)
        }


# ============ 同步封装 ============

def create_sync_client(cloud_config: CloudTrainingConfig) -> CloudAPIClient:
    """创建同步API客户端"""
    return CloudAPIClient(cloud_config)


async def generate_samples_sync(
    cloud_config: CloudTrainingConfig,
    topic: str,
    count: int = 10
) -> List[GeneratedSample]:
    """同步生成样本"""
    client = CloudAPIClient(cloud_config)
    return await client.generate_samples(topic, count)
