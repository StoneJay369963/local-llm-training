"""
模型导出模块
支持导出为 ONNX、GGUF 等格式
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器"""

    def __init__(self, config):
        self.config = config

    def export(self, output_path: Path, format: str = "onnx"):
        """导出模型"""
        logger.info(f"导出模型到 {output_path}, 格式: {format}")

        if format == "onnx":
            self._export_onnx(output_path)
        elif format == "gguf":
            self._export_gguf(output_path)
        elif format == "safetensors":
            self._export_safetensors(output_path)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _export_onnx(self, output_path: Path):
        """导出为 ONNX 格式"""
        logger.warning("ONNX 导出需要 onnxruntime，可使用: pip install onnxruntime")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = self.config.finetune_model.name_or_path
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 导出
            dummy_input = tokenizer("test", return_tensors="pt")
            
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
            )
            logger.info(f"ONNX 模型已导出到 {output_path}")
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            raise

    def _export_gguf(self, output_path: Path):
        """导出为 GGUF 格式"""
        logger.warning("GGUF 导出需要 llama.cpp，可参考: https://github.com/ggerganov/llama.cpp")
        logger.info("建议使用 llama.cpp 的 convert.py 脚本进行转换")
        raise NotImplementedError("GGUF 导出需要使用 llama.cpp 工具")

    def _export_safetensors(self, output_path: Path):
        """导出为 Safetensors 格式"""
        from transformers import AutoModelForCausalLM
        import torch
        
        model_name = self.config.finetune_model.name_or_path
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 保存
        state_dict = model.state_dict()
        
        # 拆分大文件
        max_size = 5 * 1024 * 1024 * 1024  # 5GB
        current_size = 0
        part_idx = 0
        current_tensors = {}
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            current_size += tensor_size
            
            if current_size > max_size:
                # 保存当前分片
                output_part = output_path.parent / f"{output_path.stem}-0000{part_idx}{output_path.suffix}"
                from safetensors import safe_save
                safe_save(current_tensors, str(output_part))
                current_tensors = {}
                current_size = tensor_size
                part_idx += 1
            
            current_tensors[key] = tensor
        
        # 保存剩余
        if current_tensors:
            output_part = output_path.parent / f"{output_path.stem}-0000{part_idx}{output_path.suffix}"
            from safetensors import safe_save
            safe_save(current_tensors, str(output_part))
        
        logger.info(f"Safetensors 模型已导出到 {output_path}")