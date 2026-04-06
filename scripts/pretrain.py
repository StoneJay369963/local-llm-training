"""
小型LLM预训练脚本
适用于从头训练60M-500M参数规模的模型
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class TinyLLMModel:
    """小型Transformer语言模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_model(self) -> tuple:
        """构建模型和分词器"""
        model_config = self.config['model']

        # 创建模型配置
        hf_config = AutoConfig.from_pretrained(
            "gpt2",  # 使用GPT2配置作为基础
            vocab_size=model_config['vocab_size'],
            n_positions=model_config['max_position_embeddings'],
            n_embd=model_config['hidden_size'],
            n_layer=model_config['num_hidden_layers'],
            n_head=model_config['num_attention_heads'],
            intermediate_size=model_config['intermediate_size'],
            hidden_dropout_prob=model_config['dropout'],
            attention_probs_dropout_prob=model_config['dropout'],
            layer_norm_eps=model_config['layer_norm_eps'],
            initializer_range=0.02,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

        # 创建模型
        logger.info(f"创建模型: {model_config['hidden_size']} hidden, {model_config['num_hidden_layers']} layers")
        model = AutoModelForCausalLM.from_config(hf_config)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params / 1e6:.2f}M")
        logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")

        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer


class PreTrainer:
    """预训练器"""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def prepare_dataset(self) -> tuple:
        """准备数据集"""
        dataset_config = self.config['dataset']

        # 加载数据
        data_files = {}
        if os.path.exists(dataset_config['train_path']):
            data_files['train'] = dataset_config['train_path']
        if os.path.exists(dataset_config.get('eval_path', '')):
            data_files['eval'] = dataset_config['eval_path']

        if not data_files:
            # 使用示例数据集
            logger.info("使用WikiText2作为示例数据集")
            raw_datasets = load_dataset("wikitext", "wikitext-2-v1")
        else:
            logger.info(f"加载数据文件: {data_files}")
            raw_datasets = load_dataset(
                "text",
                data_files=data_files,
                cache_dir="./data_cache"
            )

        def tokenize_function(examples):
            """分词函数"""
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config['tokenizer']['max_length'],
                padding='max_length',
            )
            result['labels'] = result['input_ids'].copy()
            return result

        # 分词
        with transformers.logging.set_verbosity_error():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.config['hardware'].get('num_workers', 4),
                remove_columns=['text'],
                load_from_cache_file=not dataset_config.get('overwrite_cache', True),
                desc="分词处理"
            )

        return tokenized_datasets

    def create_training_args(self) -> TrainingArguments:
        """创建训练参数"""
        train_config = self.config['training']
        logging_config = self.config['logging']

        output_dir = train_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_config['log_dir'], exist_ok=True)

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config['num_train_epochs'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            adam_beta1=train_config['adam_beta1'],
            adam_beta2=train_config['adam_beta2'],
            adam_epsilon=train_config['adam_epsilon'],
            max_grad_norm=train_config['max_grad_norm'],
            warmup_ratio=train_config['warmup_ratio'],
            warmup_steps=train_config['warmup_steps'],
            logging_dir=logging_config['log_dir'],
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config['eval_steps'],
            save_total_limit=train_config['save_total_limit'],
            fp16=self.config['hardware'].get('device') == 'cuda' and train_config['fp16'],
            bf16=train_config['bf16'],
            gradient_checkpointing=train_config['gradient_checkpointing'],
            optim=train_config['optim'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            seed=train_config['seed'],
            remove_unused_columns=train_config['remove_unused_columns'],
            report_to=["tensorboard"],
        )

    def train(self):
        """执行预训练"""
        # 准备数据集
        datasets = self.prepare_dataset()

        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 因果语言模型不使用MLM
        )

        # 创建训练参数
        training_args = self.create_training_args()

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=datasets['train'] if 'train' in datasets else None,
            eval_dataset=datasets.get('eval'),
            tokenizer=self.tokenizer,
        )

        # 开始训练
        logger.info("开始预训练...")
        train_result = trainer.train()

        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        trainer.save_state()

        # 记录训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return trainer


class InferenceTester:
    """推理测试器"""

    def __init__(self, model_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载模型
        logger.info(f"从 {model_path} 加载模型")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    """主函数"""
    # 解析参数
    config_path = "config/pretrain_config.yaml"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config_manager = ConfigManager(config_path)

    # 设置随机种子
    set_seed(config_manager.get('training.seed', 42))

    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("未检测到GPU，将使用CPU训练（非常慢）")

    # 构建模型
    model_builder = TinyLLMModel(config_manager.config)
    model, tokenizer = model_builder.build_model()

    # 创建训练器
    trainer = PreTrainer(model, tokenizer, config_manager.config)

    # 执行训练
    trainer.train()

    # 测试推理
    logger.info("\n测试推理:")
    tester = InferenceTester(
        config_manager.get('training.output_dir'),
        tokenizer
    )

    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "Artificial intelligence is"
    ]

    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        output = tester.generate(prompt, max_new_tokens=50)
        logger.info(f"Generated: {output}")


if __name__ == "__main__":
    main()
