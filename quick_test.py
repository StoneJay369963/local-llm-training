"""
LocalLLMTraining 快速测试脚本
自动下载小模型并测试基本功能
"""

import os
import sys
import warnings

# 禁用版本检查警告
os.environ["DISABLE_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore")

from pathlib import Path
from core.config import AppConfig, ConfigFactory
from core.inference import InferenceEngine

def main():
    print("=" * 60)
    print("LocalLLMTraining 快速测试")
    print("=" * 60)
    print()

    # 1. 测试配置
    print("[1/4] 测试配置系统...")
    try:
        config = ConfigFactory.create_finetune_config("microsoft/phi-2")
        print(f"  ✓ 配置创建成功")
        print(f"    训练模式: {config.training.mode}")
        print(f"    模型: {config.finetune_model.name_or_path}")
    except Exception as e:
        print(f"  ✗ 配置失败: {e}")
        return

    # 2. 测试推理引擎初始化
    print()
    print("[2/4] 初始化推理引擎 (正在下载模型，约500MB)...")
    print("  提示: 首次运行需要下载模型，请耐心等待...")
    try:
        # 使用较小的模型
        engine = InferenceEngine(
            config=config,
            model_path="microsoft/phi-2",
            load_in_8bit=True,  # 使用8bit减少显存
        )
        print(f"  ✓ 推理引擎初始化成功")
        print(f"    模型: microsoft/phi-2")
    except Exception as e:
        print(f"  ✗ 引擎初始化失败: {e}")
        print()
        print("  如果是显存问题，尝试使用 CPU 模式:")
        print("  编辑 config.yaml，设置 hardware.device: cpu")
        return

    # 3. 测试生成
    print()
    print("[3/4] 测试文本生成...")
    try:
        result = engine.generate(
            prompt="Hello, how are you today?",
            max_new_tokens=50,
            temperature=0.7,
        )
        print(f"  ✓ 生成成功!")
        print(f"  输出: {result[:100]}...")
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        return

    # 4. 保存测试配置
    print()
    print("[4/4] 保存测试配置...")
    try:
        config_path = Path("config_test.yaml")
        config.save(config_path)
        print(f"  ✓ 配置已保存到: {config_path}")
    except Exception as e:
        print(f"  ✗ 保存失败: {e}")

    print()
    print("=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
    print()
    print("下一步:")
    print("  1. 查看测试配置: config_test.yaml")
    print("  2. 启动训练: python -m cli.interface train config_test.yaml")
    print("  3. 启动服务: python -m cli.interface serve config_test.yaml")


if __name__ == "__main__":
    main()
