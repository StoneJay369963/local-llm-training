"""
命令行交互界面模块
提供交互式配置向导和命令接口
"""

import sys
import click
from pathlib import Path
from typing import Optional, Dict, Any, List
import inquirer
from inquirer import themes
import logging

from core.config import (
    AppConfig, ConfigFactory, TrainingMode, ModelType, DataFormat,
    PathConfig, HardwareConfig, ModelArchitectureConfig, LoRAConfig,
    QuantizationConfig, OptimizerConfig, TrainingConfig, DatasetConfig,
    TokenizerConfig, CloudTrainingConfig
)
from core.trainer import PretrainRunner, FinetuneRunner
from core.inference import InferenceEngine
from core.api_client import CloudAPIClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Color:
    """终端颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """打印标题"""
    click.echo(f"\n{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}")
    click.echo(f"{Color.BOLD}{Color.CYAN}{text:^60}{Color.END}")
    click.echo(f"{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}\n")


def print_success(text: str):
    """打印成功信息"""
    click.echo(f"{Color.GREEN}✓ {text}{Color.END}")


def print_error(text: str):
    """打印错误信息"""
    click.echo(f"{Color.RED}✗ {text}{Color.END}")


def print_warning(text: str):
    """打印警告信息"""
    click.echo(f"{Color.YELLOW}⚠ {text}{Color.END}")


def print_info(text: str):
    """打印信息"""
    click.echo(f"{Color.BLUE}ℹ {text}{Color.END}")


class InteractiveSetup:
    """交互式设置向导"""

    def __init__(self):
        self.config = AppConfig()
        self.project_root = Path.cwd()

    def run_setup(self) -> AppConfig:
        """运行完整的设置流程"""
        print_header("本地LLM训练框架 - 交互式设置向导")

        click.echo("欢迎使用本地LLM训练框架！")
        click.echo("这个向导将帮助你完成初始配置。\n")

        # 步骤1: 项目路径配置
        self._setup_paths()

        # 步骤2: 硬件配置
        self._setup_hardware()

        # 步骤3: 选择训练模式
        self._setup_training_mode()

        # 步骤4: 模型配置
        self._setup_model()

        # 步骤5: 数据配置
        self._setup_data()

        # 步骤6: 优化器配置
        self._setup_optimizer()

        # 步骤7: 云端训练配置
        self._setup_cloud_training()

        # 步骤8: 保存配置
        self._save_config()

        return self.config

    def _setup_paths(self):
        """设置路径配置"""
        print_header("步骤 1: 路径配置")

        questions = [
            inquirer.Path(
                'project_root',
                message="项目根目录",
                default=str(self.project_root),
                path_type=inquirer.Path.DIRECTORY,
                exists=True
            ),
            inquirer.Path(
                'data_dir',
                message="数据目录",
                default="data",
                path_type=inquirer.Path.DIRECTORY
            ),
            inquirer.Path(
                'model_dir',
                message="模型目录",
                default="models",
                path_type=inquirer.Path.DIRECTORY
            ),
        ]

        answers = inquirer.prompt(questions)

        self.config.paths = PathConfig(
            project_root=Path(answers['project_root']),
            data_dir=Path(answers['data_dir']),
            model_dir=Path(answers['model_dir']),
            log_dir=Path("logs"),
            cache_dir=Path("cache"),
            output_dir=Path("output")
        )

        print_success("路径配置完成")

    def _setup_hardware(self):
        """设置硬件配置"""
        print_header("步骤 2: 硬件配置")

        # 检测GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print_info(f"检测到GPU: {gpu_name}")
                print_info(f"显存: {gpu_memory:.1f} GB")

                # 根据显存推荐配置
                if gpu_memory < 10:
                    print_warning("显存较小，建议使用QLoRA和量化训练")
                    default_batch_size = 2
                    max_length = 1024
                elif gpu_memory < 24:
                    default_batch_size = 4
                    max_length = 2048
                else:
                    default_batch_size = 8
                    max_length = 4096

                default_device = "cuda"
            else:
                print_warning("未检测到GPU，将使用CPU训练（非常慢）")
                default_device = "cpu"
                default_batch_size = 1
                max_length = 512

        except ImportError:
            print_warning("PyTorch未安装，使用默认配置")
            default_device = "cuda"
            default_batch_size = 4
            max_length = 2048

        questions = [
            inquirer.List(
                'device',
                message="选择训练设备",
                choices=['cuda', 'cpu', 'mps'],
                default=default_device
            ),
            inquirer.Confirm(
                'mixed_precision',
                message="启用混合精度训练 (FP16/BF16)",
                default=True
            ),
            inquirer.Confirm(
                'gradient_checkpointing',
                message="启用梯度检查点 (以计算换显存)",
                default=True
            ),
            inquirer.Text(
                'num_workers',
                message="数据加载线程数",
                default="4"
            ),
        ]

        answers = inquirer.prompt(questions)

        self.config.hardware = HardwareConfig(
            device=answers['device'],
            mixed_precision=answers['mixed_precision'],
            gradient_checkpointing=answers['gradient_checkpointing'],
            num_workers=int(answers['num_workers'])
        )

        print_success("硬件配置完成")

    def _setup_training_mode(self):
        """设置训练模式"""
        print_header("步骤 3: 训练模式")

        questions = [
            inquirer.List(
                'mode',
                message="选择训练模式",
                choices=[
                    ('预训练 - 从零开始训练小型模型 (学习原理)', 'pretrain'),
                    ('LoRA微调 - 基于开源模型高效微调', 'lora'),
                    ('QLoRA微调 - 量化模型微调，显存占用更低', 'qlora'),
                ]
            )
        ]

        answers = inquirer.prompt(questions)

        if answers['mode'] == 'pretrain':
            self.config.training.mode = TrainingMode.PRETRAIN
        elif answers['mode'] == 'lora':
            self.config.training.mode = TrainingMode.LORA
        else:
            self.config.training.mode = TrainingMode.QLORA

        print_success(f"训练模式: {answers['mode']}")

    def _setup_model(self):
        """设置模型配置"""
        print_header("步骤 4: 模型配置")

        if self.config.training.mode == TrainingMode.PRETRAIN:
            # 预训练模型配置
            questions = [
                inquirer.Text(
                    'hidden_size',
                    message="隐藏层维度",
                    default="512",
                    validate=lambda _, x: x.isdigit() and 64 <= int(x) <= 4096
                ),
                inquirer.Text(
                    'num_hidden_layers',
                    message="Transformer层数",
                    default="8",
                    validate=lambda _, x: x.isdigit() and 1 <= int(x) <= 48
                ),
                inquirer.Text(
                    'num_attention_heads',
                    message="注意力头数",
                    default="8",
                    validate=lambda _, x: x.isdigit() and 1 <= int(x) <= 64
                ),
                inquirer.Text(
                    'vocab_size',
                    message="词表大小",
                    default="32000",
                    validate=lambda _, x: x.isdigit() and 1000 <= int(x) <= 100000
                ),
            ]

            answers = inquirer.prompt(questions)

            self.config.model = ModelArchitectureConfig(
                vocab_size=int(answers['vocab_size']),
                hidden_size=int(answers['hidden_size']),
                num_hidden_layers=int(answers['num_hidden_layers']),
                num_attention_heads=int(answers['num_attention_heads']),
                max_position_embeddings=2048,
                intermediate_size=int(answers['hidden_size']) * 4
            )

            # 计算参数量
            params = self.config.model.calculate_parameters()
            print_info(f"预估参数量: {params / 1e6:.2f}M")

        else:
            # 微调模型配置
            popular_models = [
                ('TinyLlama 1.1B (推荐入门)', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                ('Phi-1.5 1.3B', 'microsoft/phi-1_5'),
                ('GPT-2 Medium 345M', 'gpt2-medium'),
                ('Bloom 560M', 'bigscience/bloom-560m'),
                ('Qwen 1.8B', 'Qwen/Qwen-1.8B'),
                ('自定义模型', 'custom'),
            ]

            questions = [
                inquirer.List(
                    'model_choice',
                    message="选择基础模型",
                    choices=[m[0] for m in popular_models]
                ),
            ]

            answers = inquirer.prompt(questions)

            # 获取选择的模型
            selected = next(m for m in popular_models if m[0] == answers['model_choice'])

            if selected[1] == 'custom':
                model_name = click.prompt("请输入自定义模型名称或路径")
            else:
                model_name = selected[1]

            from core.config import FinetuneModelConfig
            self.config.finetune_model = FinetuneModelConfig(name_or_path=model_name)

            # LoRA配置
            if self.config.training.mode in [TrainingMode.LORA, TrainingMode.QLORA]:
                self._setup_lora()

            # 量化配置
            if self.config.training.mode == TrainingMode.QLORA:
                self._setup_quantization()

        print_success("模型配置完成")

    def _setup_lora(self):
        """设置LoRA配置"""
        questions = [
            inquirer.Text(
                'r',
                message="LoRA rank (维度)",
                default="16",
                validate=lambda _, x: x.isdigit() and 1 <= int(x) <= 128
            ),
            inquirer.Text(
                'lora_alpha',
                message="LoRA alpha (缩放因子)",
                default="32"
            ),
            inquirer.Text(
                'lora_dropout',
                message="LoRA dropout",
                default="0.05"
            ),
        ]

        answers = inquirer.prompt(questions)

        self.config.lora = LoRAConfig(
            r=int(answers['r']),
            lora_alpha=int(answers['lora_alpha']),
            lora_dropout=float(answers['lora_dropout'])
        )

    def _setup_quantization(self):
        """设置量化配置"""
        questions = [
            inquirer.List(
                'quant_type',
                message="量化精度",
                choices=[
                    ('4-bit NF4 (推荐，更低显存)', '4bit'),
                    ('8-bit (平衡)', '8bit'),
                ]
            ),
            inquirer.Confirm(
                'double_quant',
                message="启用双重量化",
                default=True
            ),
        ]

        answers = inquirer.prompt(questions)

        self.config.quantization = QuantizationConfig(
            enable=True,
            load_in_4bit=answers['quant_type'] == '4bit',
            load_in_8bit=answers['quant_type'] == '8bit',
            bnb_4bit_use_double_quant=answers['double_quant']
        )

    def _setup_data(self):
        """设置数据配置"""
        print_header("步骤 5: 数据配置")

        questions = [
            inquirer.List(
                'data_source',
                message="数据来源",
                choices=[
                    ('使用示例数据集 (WikiText)', 'example'),
                    ('指定本地文件', 'local'),
                    ('稍后配置', 'later'),
                ]
            ),
        ]

        answers = inquirer.prompt(questions)

        if answers['data_source'] == 'example':
            self.config.dataset.train_path = Path("data/train.txt")
            self.config.dataset.eval_path = Path("data/eval.txt")
            print_info("将自动下载WikiText作为示例数据集")
        elif answers['data_source'] == 'local':
            train_path = click.prompt("训练数据路径")
            eval_path = click.prompt("验证数据路径 (可选，直接回车跳过)")
            self.config.dataset.train_path = Path(train_path) if train_path else None
            self.config.dataset.eval_path = Path(eval_path) if eval_path else None

        # 数据格式
        questions = [
            inquirer.List(
                'data_format',
                message="数据格式",
                choices=[
                    ('原始文本 (每行一段文本)', 'raw'),
                    ('ChatML (对话格式)', 'chatml'),
                    ('Alpaca (指令格式)', 'alpaca'),
                ]
            ),
        ]

        answers = inquirer.prompt(questions)
        self.config.dataset.data_format = DataFormat(answers['data_format'])

        print_success("数据配置完成")

    def _setup_optimizer(self):
        """设置优化器配置"""
        print_header("步骤 6: 优化器配置")

        questions = [
            inquirer.Text(
                'learning_rate',
                message="学习率",
                default="1e-4"
            ),
            inquirer.Text(
                'num_epochs',
                message="训练轮数",
                default="3"
            ),
            inquirer.Text(
                'batch_size',
                message="Batch size",
                default="4"
            ),
            inquirer.Text(
                'gradient_accumulation',
                message="梯度累积步数",
                default="4"
            ),
        ]

        answers = inquirer.prompt(questions)

        self.config.optimizer = OptimizerConfig(
            learning_rate=float(answers['learning_rate']),
        )

        self.config.training = TrainingConfig(
            num_train_epochs=int(answers['num_epochs']),
            per_device_train_batch_size=int(answers['batch_size']),
            per_device_eval_batch_size=int(answers['batch_size']),
            gradient_accumulation_steps=int(answers['gradient_accumulation']),
        )

        print_success("优化器配置完成")

    def _setup_cloud_training(self):
        """设置云端训练配置"""
        print_header("步骤 7: 云端训练配置 (可选)")

        questions = [
            inquirer.Confirm(
                'enable_cloud',
                message="启用云端API辅助训练 (生成合成数据/质量过滤)",
                default=False
            ),
        ]

        answers = inquirer.prompt(questions)

        if answers['enable_cloud']:
            self._setup_api_provider()

        print_success("云端训练配置完成")

    def _setup_api_provider(self):
        """设置API提供商"""
        questions = [
            inquirer.List(
                'provider',
                message="选择API提供商",
                choices=[
                    ('OpenAI (GPT-3.5/GPT-4)', 'openai'),
                    ('自定义API', 'custom'),
                ]
            ),
            inquirer.Text(
                'api_key',
                message="API密钥 (可选，支持环境变量 OPENAI_API_KEY)",
                default=""
            ),
            inquirer.Text(
                'model_name',
                message="模型名称",
                default="gpt-3.5-turbo"
            ),
        ]

        answers = inquirer.prompt(questions)

        from core.config import APIConfig, CloudTrainingConfig as CTC

        api_config = APIConfig(
            provider=answers['provider'],
            api_key=answers['api_key'] or None,
            model_name=answers['model_name']
        )

        self.config.cloud_training = CTC(
            enable=True,
            api=api_config,
            enable_data_augmentation=True,
            enable_quality_filter=True,
            generate_synthetic_data=True
        )

    def _save_config(self):
        """保存配置"""
        print_header("步骤 8: 保存配置")

        default_path = self.config.paths.project_root / "config.yaml"

        questions = [
            inquirer.Confirm(
                'save',
                message=f"保存配置到 {default_path}",
                default=True
            ),
        ]

        answers = inquirer.prompt(questions)

        if answers['save']:
            self.config.ensure_directories()
            self.config.save(default_path)
            print_success(f"配置已保存到 {default_path}")
        else:
            custom_path = click.prompt("请输入保存路径")
            self.config.save(Path(custom_path))
            print_success(f"配置已保存到 {custom_path}")


class CLIInterface:
    """命令行界面"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """加载配置"""
        if self.config_path and self.config_path.exists():
            return AppConfig.load(self.config_path)
        return AppConfig()

    # ============ 命令组 ============

    @staticmethod
    def init_command():
        """初始化命令"""
        setup = InteractiveSetup()
        config = setup.run_setup()
        return config

    @staticmethod
    def train_command(config_path: Path, mode: Optional[str] = None):
        """训练命令"""
        config = AppConfig.load(config_path)

        if mode:
            config.training.mode = TrainingMode(mode)

        if config.training.mode == TrainingMode.PRETRAIN:
            runner = PretrainRunner(config)
            runner.train()
        else:
            runner = FinetuneRunner(config)
            runner.train()

    @staticmethod
    def infer_command(
        config_path: Path,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        """推理命令"""
        config = AppConfig.load(config_path)
        config.inference.max_new_tokens = max_new_tokens
        config.inference.temperature = temperature

        engine = InferenceEngine(config)
        result = engine.generate(prompt)
        click.echo(f"\n{Color.GREEN}模型输出:{Color.END}\n{result}")

    @staticmethod
    def eval_command(config_path: Path, dataset_path: Path):
        """评估命令"""
        config = AppConfig.load(config_path)
        from core.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate(dataset_path)

        click.echo(f"\n{Color.CYAN}评估结果:{Color.END}")
        for key, value in metrics.items():
            click.echo(f"  {key}: {value}")

    @staticmethod
    def export_command(
        config_path: Path,
        output_path: Path,
        format: str = "onnx"
    ):
        """导出命令"""
        config = AppConfig.load(config_path)
        from core.exporter import ModelExporter

        exporter = ModelExporter(config)
        exporter.export(output_path, format=format)
        print_success(f"模型已导出到 {output_path}")

    @staticmethod
    def serve_command(config_path: Path, host: str = "0.0.0.0", port: int = 8000):
        """启动Web服务"""
        config = AppConfig.load(config_path)
        from api.server import create_app

        app = create_app(config)
        print_info(f"启动Web服务: http://{host}:{port}")
        print_info("按 Ctrl+C 停止服务")

        app.run(host=host, port=port, debug=False)


# ============ Click命令定义 ============

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """本地LLM训练框架 - 命令行工具"""
    pass


@cli.command()
def init():
    """交互式初始化配置"""
    config = CLIInterface.init_command()
    print_success("设置完成！")


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['pretrain', 'finetune', 'lora', 'qlora']),
              help='覆盖训练模式')
def train(config_path: str, mode: Optional[str]):
    """训练模型"""
    CLIInterface.train_command(Path(config_path), mode)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('prompt')
@click.option('--max-tokens', default=512, help='最大生成长度')
@click.option('--temperature', default=0.7, help='生成温度')
def infer(config_path: str, prompt: str, max_tokens: int, temperature: float):
    """运行推理"""
    CLIInterface.infer_command(
        Path(config_path),
        prompt,
        max_tokens,
        temperature
    )


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
def eval(config_path: str, dataset_path: str):
    """评估模型"""
    CLIInterface.eval_command(Path(config_path), Path(dataset_path))


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--format', default='onnx', help='导出格式')
def export(config_path: str, output_path: str, format: str):
    """导出模型"""
    CLIInterface.export_command(Path(config_path), Path(output_path), format)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='服务地址')
@click.option('--port', default=8000, help='服务端口')
def serve(config_path: str, host: str, port: int):
    """启动Web服务"""
    CLIInterface.serve_command(Path(config_path), host, port)


@cli.command()
def info():
    """显示系统信息"""
    print_header("系统信息")

    # Python版本
    import sys
    click.echo(f"Python: {sys.version}")

    # PyTorch版本
    try:
        import torch
        click.echo(f"PyTorch: {torch.__version__}")
        click.echo(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            click.echo(f"CUDA版本: {torch.version.cuda}")
            click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print_warning("PyTorch未安装")

    # Transformers版本
    try:
        import transformers
        click.echo(f"Transformers: {transformers.__version__}")
    except ImportError:
        print_warning("Transformers未安装")


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
def model_info(model_path: str):
    """显示模型信息并检测格式"""
    from core.model_loader import ModelLoader, MODEL_FORMAT_INFO

    print_header("模型信息")

    loader = ModelLoader()
    model_path = Path(model_path)

    # 检测格式
    fmt = loader.detect_format(model_path)
    click.echo(f"\n模型路径: {model_path}")
    click.echo(f"检测到的格式: {Color.CYAN}{fmt.value}{Color.END}")

    # 显示格式详情
    if fmt in MODEL_FORMAT_INFO:
        info = MODEL_FORMAT_INFO[fmt]
        click.echo(f"\n格式名称: {info['name']}")
        click.echo(f"描述: {info['description']}")
        click.echo(f"量化: {'是' if info['quantized'] else '否'}")
        click.echo(f"加载器: {info['loader']}")
        click.echo(f"\n优点: {', '.join(info['pros'])}")
        click.echo(f"缺点: {', '.join(info['cons'])}")

    # 列出目录中的文件
    if model_path.is_dir():
        click.echo(f"\n目录内容:")
        files = list(model_path.glob("*"))
        for f in sorted(files)[:20]:  # 只显示前20个
            size = f.stat().st_size
            if size > 1024 * 1024 * 1024:
                size_str = f"{size / 1024 / 1024 / 1024:.2f} GB"
            elif size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} B"
            click.echo(f"  {f.name}: {size_str}")

        if len(files) > 20:
            click.echo(f"  ... 还有 {len(files) - 20} 个文件")

    print()


@cli.command()
def formats():
    """显示支持的模型格式"""
    from core.model_loader import MODEL_FORMAT_INFO

    print_header("支持的模型格式")

    for fmt, info in MODEL_FORMAT_INFO.items():
        if fmt == "auto":
            continue

        click.echo(f"\n{Color.BOLD}[{info['name']}]{Color.END} ({fmt})")
        click.echo(f"  描述: {info['description']}")
        click.echo(f"  扩展名: {', '.join(info['extension'])}")
        click.echo(f"  量化: {'是' if info['quantized'] else '否'}")
        click.echo(f"  优点: {', '.join(info['pros'])}")

    click.echo("\n" + "=" * 60)
    click.echo("\n安装额外依赖:")
    click.echo(f"  {Color.CYAN}GGUF格式:{Color.END} pip install llama-cpp-python")
    click.echo(f"  {Color.CYAN}GPTQ格式:{Color.END} pip install auto-gptq")
    click.echo(f"  {Color.CYAN}AWQ格式:{Color.END} pip install autoawq")
    print()


if __name__ == '__main__':
    cli()
