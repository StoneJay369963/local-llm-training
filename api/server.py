"""
FastAPI Web服务模块
提供REST API接口，支持远程训练控制和状态监控
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
import logging
import asyncio
import os

from core.config import AppConfig, TrainingMode
from core.trainer import PretrainRunner, FinetuneRunner
from core.inference import InferenceEngine
from core.api_client import CloudAPIClient, DataAugmentationPipeline

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ 请求/响应模型 ============

class ConfigUpdate(BaseModel):
    """配置更新"""
    model_name: Optional[str] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    max_length: Optional[int] = None


class TrainingRequest(BaseModel):
    """训练请求"""
    config_path: Optional[str] = None
    mode: str = "finetune"
    model_name: Optional[str] = None
    data_path: Optional[str] = None


class GenerationRequest(BaseModel):
    """生成请求"""
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class DataAugmentRequest(BaseModel):
    """数据增强请求"""
    input_path: str
    output_path: str
    augment_ratio: float = 0.3
    filter_threshold: float = 7.0


class SyntheticDataRequest(BaseModel):
    """合成数据生成请求"""
    topics: List[str]
    output_path: str
    samples_per_topic: int = 100


class StatusResponse(BaseModel):
    """状态响应"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# ============ 全局状态 ============

class TrainingState:
    """训练状态管理"""
    def __init__(self):
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.elapsed_time = 0
        self.logs: List[str] = []

    def update(self, **kwargs):
        """更新状态"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self):
        """重置状态"""
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.elapsed_time = 0


training_state = TrainingState()


# ============ 创建应用 ============

def create_app(config: AppConfig) -> FastAPI:
    """创建FastAPI应用"""

    app = FastAPI(
        title="LocalLLMTraining API",
        description="本地LLM训练框架API服务",
        version="1.0.0"
    )

    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件
    if Path("web").exists():
        app.mount("/static", StaticFiles(directory="web"), name="static")

    # ============ 首页 ============

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """首页"""
        html_path = Path("templates/index.html")
        if html_path.exists():
            return html_path.read_text(encoding='utf-8')
        return """
        <html>
            <head><title>LocalLLMTraining</title></head>
            <body>
                <h1>LocalLLMTraining API</h1>
                <p>API服务已启动</p>
                <ul>
                    <li><a href="/docs">API文档</a></li>
                    <li><a href="/status">训练状态</a></li>
                    <li><a href="/health">健康检查</a></li>
                </ul>
            </body>
        </html>
        """

    # ============ 健康检查 ============

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "training_active": training_state.is_training
        }

    # ============ 配置API ============

    @app.get("/api/config")
    async def get_config():
        """获取当前配置"""
        return {
            "model": config.model.dict() if config.model else None,
            "training": config.training.dict(),
            "optimizer": config.optimizer.dict(),
            "hardware": config.hardware.dict(),
        }

    @app.post("/api/config")
    async def update_config(update: ConfigUpdate):
        """更新配置"""
        if update.model_name:
            config.finetune_model.name_or_path = update.model_name
        if update.learning_rate:
            config.optimizer.learning_rate = update.learning_rate
        if update.batch_size:
            config.training.per_device_train_batch_size = update.batch_size
        if update.num_epochs:
            config.training.num_train_epochs = update.num_epochs
        if update.max_length:
            config.tokenizer.max_length = update.max_length

        return {"status": "success", "message": "配置已更新"}

    # ============ 训练API ============

    @app.post("/api/train")
    async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
        """启动训练"""
        if training_state.is_training:
            raise HTTPException(status_code=400, detail="训练已在进行中")

        # 更新配置
        if request.config_path:
            config = AppConfig.load(Path(request.config_path))
        if request.model_name:
            config.finetune_model.name_or_path = request.model_name

        # 后台启动训练
        async def train_task():
            training_state.is_training = True
            training_state.reset()

            try:
                if config.training.mode == TrainingMode.PRETRAIN:
                    runner = PretrainRunner(config)
                else:
                    runner = FinetuneRunner(config)

                await runner.train_async(
                    progress_callback=lambda step, total, loss: training_state.update(
                        current_step=step,
                        total_steps=total,
                        current_loss=loss
                    )
                )

            except Exception as e:
                logger.error(f"训练失败: {e}")
                training_state.logs.append(f"错误: {str(e)}")
            finally:
                training_state.is_training = False

        background_tasks.add_task(train_task)

        return {
            "status": "started",
            "message": "训练已启动",
            "mode": config.training.mode
        }

    @app.post("/api/train/stop")
    async def stop_training():
        """停止训练"""
        if not training_state.is_training:
            raise HTTPException(status_code=400, detail="没有正在进行的训练")

        # TODO: 实现训练停止逻辑
        training_state.is_training = False

        return {"status": "success", "message": "训练已停止"}

    @app.get("/api/train/status")
    async def get_training_status():
        """获取训练状态"""
        return {
            "is_training": training_state.is_training,
            "current_step": training_state.current_step,
            "total_steps": training_state.total_steps,
            "current_loss": training_state.current_loss,
            "progress": (
                training_state.current_step / training_state.total_steps * 100
                if training_state.total_steps > 0 else 0
            )
        }

    @app.get("/api/train/logs")
    async def get_training_logs(lines: int = 100):
        """获取训练日志"""
        return {"logs": training_state.logs[-lines:]}

    # ============ 推理API ============

    @app.post("/api/generate")
    async def generate_text(request: GenerationRequest):
        """生成文本"""
        try:
            engine = InferenceEngine(config)
            result = engine.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )

            return {
                "status": "success",
                "prompt": request.prompt,
                "generated": result
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ 数据处理API ============

    @app.post("/api/data/augment")
    async def augment_data(request: DataAugmentRequest, background_tasks: BackgroundTasks):
        """数据增强"""
        if not config.cloud_training.enable:
            raise HTTPException(
                status_code=400,
                detail="云端训练未启用，请在配置中启用CloudTraining"
            )

        async def augment_task():
            client = CloudAPIClient(config.cloud_training)
            pipeline = DataAugmentationPipeline(client)

            result = await pipeline.process_dataset(
                input_path=Path(request.input_path),
                output_path=Path(request.output_path),
                augment_ratio=request.augment_ratio,
                filter_threshold=request.filter_threshold
            )

            training_state.logs.append(f"数据增强完成: {result}")

        background_tasks.add_task(augment_task)

        return {
            "status": "started",
            "message": "数据增强任务已提交"
        }

    @app.post("/api/data/synthetic")
    async def generate_synthetic_data(
        request: SyntheticDataRequest,
        background_tasks: BackgroundTasks
    ):
        """生成合成数据"""
        if not config.cloud_training.enable:
            raise HTTPException(
                status_code=400,
                detail="云端训练未启用"
            )

        async def generate_task():
            client = CloudAPIClient(config.cloud_training)
            pipeline = DataAugmentationPipeline(client)

            result = await pipeline.generate_synthetic_dataset(
                topics=request.topics,
                output_path=Path(request.output_path),
                samples_per_topic=request.samples_per_topic
            )

            training_state.logs.append(f"合成数据生成完成: {result}")

        background_tasks.add_task(generate_task)

        return {
            "status": "started",
            "message": "合成数据生成任务已提交"
        }

    @app.post("/api/data/upload")
    async def upload_data(file: UploadFile = File(...)):
        """上传数据文件"""
        upload_dir = config.paths.data_dir
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename

        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content)
        }

    # ============ 模型管理API ============

    @app.get("/api/models")
    async def list_models():
        """列出可用模型"""
        model_dir = config.paths.model_dir

        if not model_dir.exists():
            return {"models": []}

        models = []
        for item in model_dir.iterdir():
            if item.is_dir():
                # 检查是否为有效模型目录
                if (item / "config.json").exists() or (item / "adapter_config.json").exists():
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "size": sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    })

        return {"models": models}

    @app.get("/api/models/{model_name}")
    async def get_model_info(model_name: str):
        """获取模型信息"""
        model_path = config.paths.model_dir / model_name

        if not model_path.exists():
            raise HTTPException(status_code=404, detail="模型不存在")

        # TODO: 读取模型配置信息
        return {
            "name": model_name,
            "path": str(model_path)
        }

    # ============ 系统信息API ============

    @app.get("/api/system/info")
    async def get_system_info():
        """获取系统信息"""
        info = {
            "python_version": os.sys.version,
            "platform": os.name,
        }

        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

        return info

    return app


# ============ 独立运行 ============

def run_server(config: AppConfig, host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import uvicorn
    from core.config import AppConfig

    config = AppConfig()
    app = create_app(config)

    print("启动API服务: http://0.0.0.0:8000")
    print("API文档: http://0.0.0.0:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
