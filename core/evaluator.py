"""
模型评估模块
提供困惑度、BLEU、ROUGE 等指标计算
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def _setup_model(self):
        """加载模型"""
        if self.config.finetune_model:
            model_name = self.config.finetune_model.name_or_path
        else:
            raise ValueError("未配置模型")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.finetune_model.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=self.config.finetune_model.trust_remote_code
        )

    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        """评估模型"""
        if not self.model:
            self._setup_model()

        logger.info(f"评估数据集: {dataset_path}")

        # 计算困惑度
        perplexity = self._calculate_perplexity(dataset_path)

        return {
            "perplexity": perplexity,
            "loss": perplexity ** -1,
        }

    def _calculate_perplexity(self, dataset_path: Path) -> float:
        """计算困惑度"""
        from math import exp

        if not dataset_path.exists():
            logger.warning(f"数据集不存在: {dataset_path}")
            return float('inf')

        # 加载数据
        ext = dataset_path.suffix.lower()
        if ext == '.json':
            dataset = load_dataset("json", data_files={"test": str(dataset_path)})["test"]
        else:
            dataset = load_dataset("text", data_files={"test": str(dataset_path)})["test"]

        total_loss = 0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in dataset["text"]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.tokenizer.max_length
                ).to(self.model.device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = exp(avg_loss)

        return perplexity