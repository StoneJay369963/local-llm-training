"""
训练引擎模块
支持预训练、LoRA微调、QLoRA微调
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import json

import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    set_seed,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, Dataset

from core.config import (
    AppConfig,
    TrainingMode,
    PretrainModelConfig,
    FinetuneModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    elapsed_time: float
    throughput: float  # samples/second


class BaseTrainer:
    """训练器基类"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """获取训练设备"""
        if self.config.hardware.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.config.hardware.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_model(self):
        """设置模型"""
        raise NotImplementedError

    def _setup_tokenizer(self):
        """设置分词器"""
        raise NotImplementedError

    def _setup_training(self):
        """设置训练"""
        raise NotImplementedError

    def train(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """执行训练"""
        raise NotImplementedError

    async def train_async(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """异步执行训练"""
        raise NotImplementedError

    def save_model(self, output_dir: Optional[Path] = None):
        """保存模型"""
        if self.trainer:
            output_dir = output_dir or self.config.training.output_dir
            self.trainer.save_model(str(output_dir))
            logger.info(f"模型已保存到 {output_dir}")


class PretrainRunner(BaseTrainer):
    """预训练运行器"""

    def __init__(self, config: AppConfig):
        super().__init__(config)

        if not config.model:
            raise ValueError("预训练需要提供模型配置")

        self._setup_model()
        self._setup_tokenizer()
        self._setup_training()

    def _setup_model(self):
        """设置预训练模型"""
        model_config = self.config.model
        assert isinstance(model_config, PretrainModelConfig)

        logger.info("创建预训练模型...")

        # 使用GPT2配置作为基础创建自定义模型
        from transformers import GPT2Config

        hf_config = GPT2Config(
            vocab_size=model_config.vocab_size,
            n_positions=model_config.max_position_embeddings,
            n_embd=model_config.hidden_size,
            n_layer=model_config.num_hidden_layers,
            n_head=model_config.num_attention_heads,
            intermediate_size=model_config.intermediate_size,
            hidden_dropout_prob=model_config.dropout,
            attn_pdrop=model_config.dropout,
            embd_pdrop=model_config.dropout,
            layer_norm_epsilon=model_config.layer_norm_eps,
            initializer_range=model_config.initializer_range,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

        self.model = AutoModelForCausalLM.from_config(hf_config)
        self.model.to(self.device)

        # 打印参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params / 1e6:.2f}M")
        logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    def _setup_tokenizer(self):
        """设置分词器"""
        logger.info("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _setup_training(self):
        """设置训练"""
        train_config = self.config.training
        optimizer_config = self.config.optimizer

        # 确保输出目录存在
        train_config.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=str(train_config.output_dir),
            num_train_epochs=train_config.num_train_epochs,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            learning_rate=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            adam_beta1=optimizer_config.adam_beta1,
            adam_beta2=optimizer_config.adam_beta2,
            adam_epsilon=optimizer_config.adam_epsilon,
            max_grad_norm=optimizer_config.max_grad_norm,
            warmup_ratio=optimizer_config.warmup_ratio,
            warmup_steps=optimizer_config.warmup_steps,
            logging_steps=train_config.logging_steps,
            save_steps=train_config.save_steps,
            eval_steps=train_config.eval_steps,
            save_total_limit=train_config.save_total_limit,
            fp16=train_config.fp16,
            bf16=train_config.bf16,
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            optim=optimizer_config.optim,
            lr_scheduler_type=optimizer_config.lr_scheduler_type,
            seed=train_config.seed,
            remove_unused_columns=False,
            report_to=train_config.report_to,
            load_best_model_at_end=train_config.load_best_model_at_end,
            metric_for_best_model=train_config.metric_for_best_model,
            greater_is_better=train_config.greater_is_better,
        )

        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=None,  # 稍后设置
            eval_dataset=None,
            tokenizer=self.tokenizer,
        )

    def _prepare_dataset(self) -> Dataset:
        """准备数据集"""
        dataset_config = self.config.dataset

        # 加载数据
        if dataset_config.train_path and dataset_config.train_path.exists():
            logger.info(f"加载数据文件: {dataset_config.train_path}")
            raw_datasets = load_dataset(
                "text",
                data_files={"train": str(dataset_config.train_path)},
                cache_dir=str(self.config.paths.cache_dir)
            )
        else:
            # 使用示例数据集
            logger.info("使用WikiText作为示例数据集")
            raw_datasets = load_dataset("wikitext", "wikitext-2-v1")

        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.tokenizer.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        # 分词
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.hardware.num_workers,
            remove_columns=["text"],
        )

        return tokenized_datasets["train"]

    def train(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """执行预训练"""
        set_seed(self.config.training.seed)

        # 准备数据集
        train_dataset = self._prepare_dataset()
        self.trainer.train_dataset = train_dataset

        # 开始训练
        logger.info("开始预训练...")
        start_time = time.time()

        train_result = self.trainer.train()

        # 记录训练时间
        elapsed = time.time() - start_time
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")

        # 保存模型
        self.save_model()

        return train_result

    async def train_async(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """异步执行预训练"""
        import asyncio

        # 在线程池中运行训练
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.train, progress_callback)


class FinetuneRunner(BaseTrainer):
    """微调运行器"""

    def __init__(self, config: AppConfig):
        super().__init__(config)

        self._setup_model()
        self._setup_tokenizer()
        self._setup_training()

    def _setup_model(self):
        """设置微调模型"""
        model_config = self.config.finetune_model
        lora_config = self.config.lora
        quant_config = self.config.quantization

        logger.info(f"加载基础模型: {model_config.name_or_path}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=model_config.use_fast,
        )

        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        model_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
            "device_map": "auto",
        }

        # 量化配置
        if quant_config.enable:
            from transformers import BitsAndBytesConfig

            if quant_config.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(
                        torch, quant_config.bnb_4bit_compute_dtype
                    ),
                    bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.name_or_path,
                **model_kwargs
            )
        except Exception as e:
            logger.warning(f"模型加载失败，使用fp16重试: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=model_config.trust_remote_code,
            )

        # 应用LoRA
        if lora_config.enable:
            logger.info(f"应用LoRA: rank={lora_config.r}, alpha={lora_config.lora_alpha}")

            if quant_config.enable:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.config.hardware.gradient_checkpointing
                )

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def _setup_tokenizer(self):
        """设置分词器"""
        self.tokenizer.padding_side = self.config.tokenizer.padding_side
        self.tokenizer.truncation_side = self.config.tokenizer.truncation_side

    def _setup_training(self):
        """设置训练"""
        train_config = self.config.training
        optimizer_config = self.config.optimizer

        # 确保输出目录存在
        train_config.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=str(train_config.output_dir),
            num_train_epochs=train_config.num_train_epochs,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            learning_rate=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            adam_beta1=optimizer_config.adam_beta1,
            adam_beta2=optimizer_config.adam_beta2,
            adam_epsilon=optimizer_config.adam_epsilon,
            max_grad_norm=optimizer_config.max_grad_norm,
            warmup_ratio=optimizer_config.warmup_ratio,
            warmup_steps=optimizer_config.warmup_steps,
            logging_steps=train_config.logging_steps,
            save_steps=train_config.save_steps,
            eval_steps=train_config.eval_steps,
            save_total_limit=train_config.save_total_limit,
            fp16=train_config.fp16,
            bf16=train_config.bf16,
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            optim=optimizer_config.optim,
            lr_scheduler_type=optimizer_config.lr_scheduler_type,
            group_by_length=train_config.group_by_length,
            seed=train_config.seed,
            remove_unused_columns=False,
            report_to=train_config.report_to,
            load_best_model_at_end=train_config.load_best_model_at_end,
            metric_for_best_model=train_config.metric_for_best_model,
            greater_is_better=train_config.greater_is_better,
        )

        # 创建数据整理器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer,
        )

    def _prepare_dataset(self) -> tuple:
        """准备数据集"""
        dataset_config = self.config.dataset

        # 加载数据
        if dataset_config.train_path and dataset_config.train_path.exists():
            logger.info(f"加载数据文件: {dataset_config.train_path}")

            ext = dataset_config.train_path.suffix.lower()
            if ext == '.json':
                raw_datasets = load_dataset("json", data_files={"train": str(dataset_config.train_path)})
            else:
                raw_datasets = load_dataset("text", data_files={"train": str(dataset_config.train_path)})
        else:
            # 使用示例数据集
            logger.info("使用Alpaca数据集")
            raw_datasets = load_dataset("yahma/alpaca-cleaned", split="train")

        # 格式化数据
        def format_function(examples):
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                    for inst, resp in zip(examples["instruction"], examples["output"])
                ]
            else:
                texts = examples[list(examples.keys())[0]]

            # Tokenize
            result = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.tokenizer.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        # 分词
        tokenized_datasets = raw_datasets.map(
            format_function,
            batched=True,
            num_proc=self.config.hardware.num_workers,
            remove_columns=raw_datasets.column_names,
        )

        # 划分训练集和验证集
        if "validation" in tokenized_datasets:
            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets["validation"]
        else:
            split = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]

        return train_dataset, eval_dataset

    def train(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """执行微调"""
        set_seed(self.config.training.seed)

        # 准备数据集
        train_dataset, eval_dataset = self._prepare_dataset()
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = eval_dataset

        # 开始训练
        logger.info("开始微调...")
        start_time = time.time()

        train_result = self.trainer.train()

        # 记录训练时间
        elapsed = time.time() - start_time
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")

        # 保存模型
        self.save_model()

        return train_result

    async def train_async(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """异步执行微调"""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.train, progress_callback)


class DistributedTrainer:
    """分布式训练器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup(self):
        """设置分布式训练"""
        if self.world_size > 1:
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
            torch.cuda.set_device(self.local_rank)

    def cleanup(self):
        """清理分布式训练"""
        if self.world_size > 1:
            torch.distributed.destroy_process_group()
