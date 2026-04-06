"""
LLM微调脚本
使用LoRA/QLoRA进行参数高效微调
支持从本地预训练模型或开源模型进行微调
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
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
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


class LoRAFinetuner:
    """LoRA微调器"""

    def __init__(
        self,
        model_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        quant_config: Dict[str, Any]
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quant_config = quant_config

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if not self.quant_config.get('enable', False):
            return None

        if self.quant_config.get('load_in_4bit', False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    self.quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
                ),
                bnb_4bit_quant_type=self.quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=self.quant_config.get('bnb_4bit_use_double_quant', True),
            )
        elif self.quant_config.get('load_in_8bit', False):
            return BitsAndBytesConfig(load_in_8bit=True)

        return None

    def load_model_and_tokenizer(self) -> tuple:
        """加载模型和分词器"""
        model_name = self.model_config['name_or_path']
        logger.info(f"加载模型: {model_name}")

        # 量化配置
        bnb_config = self._get_quantization_config()

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=self.model_config.get('use_fast', True)
        )
        tokenizer.padding_side = self.model_config.get('padding_side', 'right')
        tokenizer.truncation_side = self.model_config.get('truncation_side', 'right')

        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 加载模型
        model_kwargs = {
            "trust_remote_code": self.model_config.get('trust_remote_code', False),
            "device_map": "auto",
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # 检查是否使用缓存
        use_cache = self.model_config.get('use_cache', True)
        if not use_cache:
            model_kwargs["low_cpu_mem_usage"] = True

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("尝试使用torch_dtype加载...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        # 计算模型大小
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型总参数量: {total_params / 1e9:.2f}B")

        return model, tokenizer

    def prepare_model_for_lora(self, model) -> torch.nn.Module:
        """准备LoRA微调"""
        # 准备量化模型
        if self.quant_config.get('enable', False):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )

        # 创建LoRA配置
        lora_cfg = LoraConfig(
            r=self.lora_config['lora_rank'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            target_modules=self.lora_config['target_modules'],
            bias=self.lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(f"LoRA配置: rank={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
        logger.info(f"目标模块: {lora_cfg.target_modules}")

        # 应用LoRA
        model = get_peft_model(model, lora_cfg)

        # 打印可训练参数
        model.print_trainable_parameters()

        return model


class DatasetProcessor:
    """数据集处理器"""

    CHATML_FORMAT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""

    ALPACA_FORMAT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_chatml(self, sample: Dict) -> str:
        """ChatML格式"""
        messages = sample.get('messages', [])

        if len(messages) < 2:
            # 单轮对话
            user = sample.get('user', '')
            assistant = sample.get('assistant', '')
            system = sample.get('system', 'You are a helpful assistant.')

            prompt = self.CHATML_FORMAT.format(
                system=system,
                user=user,
                assistant=assistant
            )
        else:
            # 多轮对话
            parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt = '\n'.join(parts)

        return prompt

    def format_alpaca(self, sample: Dict) -> str:
        """Alpaca格式"""
        return self.ALPACA_FORMAT.format(
            instruction=sample.get('instruction', ''),
            input=sample.get('input', ''),
            output=sample.get('output', '')
        )

    def load_dataset(
        self,
        train_path: str,
        eval_path: Optional[str] = None,
        format_type: str = "chatml"
    ) -> tuple:
        """加载数据集"""
        def format_samples(examples):
            """格式化样本"""
            if format_type == "chatml":
                texts = [self.format_chatml(sample) for sample in zip(
                    examples.get('user', ['']),
                    examples.get('assistant', ['']),
                    examples.get('system', [''] * len(examples.get('user', [''])))
                )]
            else:
                texts = [self.format_alpaca(sample) for sample in zip(
                    examples.get('instruction', ['']),
                    examples.get('input', ['']),
                    examples.get('output', [''])
                )]

            # Tokenize
            model_inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
            )

            # 创建labels
            labels = []
            for input_ids, attention_mask in zip(
                model_inputs['input_ids'],
                model_inputs['attention_mask']
            ):
                label = input_ids.copy()
                # Mask掉user部分，只训练assistant部分
                # 这里简化处理，实际可以更精确
                labels.append(label)

            model_inputs['labels'] = labels
            return model_inputs

        # 加载数据
        data_files = {'train': train_path}
        if eval_path:
            data_files['eval'] = eval_path

        # 自动检测格式
        ext = Path(train_path).suffix.lower()
        if ext == '.json':
            dataset = load_dataset('json', data_files=data_files)['train']
        elif ext == '.txt':
            dataset = load_dataset('text', data_files=data_files)['train']
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 划分训练集和验证集
        if eval_path is None:
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = load_dataset('json', data_files={'eval': eval_path})['eval']

        # Tokenize
        train_dataset = train_dataset.map(
            format_samples,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenize训练集"
        )

        if eval_dataset:
            eval_dataset = eval_dataset.map(
                format_samples,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenize验证集"
            )

        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(eval_dataset) if eval_dataset else 0} 样本")

        return train_dataset, eval_dataset


class FinetuneRunner:
    """微调运行器"""

    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.lora_finetuner = LoRAFinetuner(
            self.config_manager.config['model'],
            self.config_manager.config['lora'],
            self.config_manager.config['quantization']
        )

    def setup_training(self):
        """设置训练环境"""
        # 设置随机种子
        set_seed(self.config_manager.get('training.seed', 42))

        # 检查GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("未检测到GPU，将使用CPU训练（非常慢）")

        # 加载模型和分词器
        model, tokenizer = self.lora_finetuner.load_model_and_tokenizer()

        # 准备LoRA
        model = self.lora_finetuner.prepare_model_for_lora(model)

        # 加载数据集
        tokenizer_cfg = self.config_manager.config['tokenizer']
        dataset_cfg = self.config_manager.config['dataset']

        dataset_processor = DatasetProcessor(
            tokenizer,
            max_length=tokenizer_cfg['max_length']
        )

        train_dataset, eval_dataset = dataset_processor.load_dataset(
            dataset_cfg['train_path'],
            dataset_cfg.get('eval_path'),
            dataset_cfg.get('format', 'chatml')
        )

        return model, tokenizer, train_dataset, eval_dataset

    def create_training_args(self) -> TrainingArguments:
        """创建训练参数"""
        train_config = self.config_manager.config['training']
        logging_config = self.config_manager.config['logging']
        hardware_config = self.config_manager.config['hardware']

        output_dir = train_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_config['log_dir'], exist_ok=True)

        # 设置日志记录
        report_to = []
        if logging_config.get('use_wandb', False):
            report_to.append("wandb")
        else:
            report_to.append("tensorboard")

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config['num_train_epochs'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            eval_accumulation_steps=train_config.get('eval_accumulation_steps'),
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
            fp16=train_config['fp16'],
            bf16=train_config['bf16'],
            gradient_checkpointing=train_config['gradient_checkpointing'],
            optim=train_config['optim'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            group_by_length=train_config.get('group_by_length', True),
            seed=train_config['seed'],
            report_to=report_to,
            load_best_model_at_end=train_config['training_args'].get('load_best_model_at_end', True),
            metric_for_best_model=train_config['training_args'].get('metric_for_best_model', 'eval_loss'),
            greater_is_better=train_config['training_args'].get('greater_is_better', False),
        )

    def train(self):
        """执行微调"""
        # 设置训练
        model, tokenizer, train_dataset, eval_dataset = self.setup_training()

        # 创建数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            padding_to_multiple_of=8 if self.config_manager.config['quantization'].get('enable') else None,
        )

        # 创建训练参数
        training_args = self.create_training_args()

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # 开始训练
        logger.info("开始LoRA微调...")
        train_result = trainer.train()

        # 保存模型
        logger.info("保存LoRA权重...")
        trainer.save_model(os.path.join(
            self.config_manager.get('training.output_dir'),
            "lora_weights"
        ))
        trainer.save_state()

        # 记录指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return trainer


class ModelExporter:
    """模型导出器"""

    @staticmethod
    def merge_lora_weights(
        base_model_path: str,
        lora_path: str,
        output_path: str
    ):
        """合并LoRA权重到基座模型"""
        from peft import PeftModel

        logger.info("加载基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info("合并LoRA权重...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()

        logger.info(f"保存合并后的模型到 {output_path}...")
        model.save_pretrained(output_path)


def main():
    """主函数"""
    # 解析参数
    config_path = "config/finetune_config.yaml"

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config_manager = ConfigManager(config_path)

    # 运行微调
    runner = FinetuneRunner(config_path)
    runner.train()

    logger.info("\n微调完成！")
    logger.info(f"模型保存在: {config_manager.get('training.output_dir')}")


if __name__ == "__main__":
    main()
