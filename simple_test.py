"""
最简单的测试脚本 - 使用 transformers 直接测试
绕过所有自定义模块
"""

import os
os.environ["DISABLE_VERSION_CHECK"] = "1"

import warnings
warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print("LocalLLMTraining 简单测试")
    print("=" * 60)
    print()

    # 1. 检查 PyTorch
    print("[1/3] 检查 PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch版本: {torch.__version__}")
        print(f"  ✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  ✗ PyTorch错误: {e}")

    # 2. 检查 Transformers
    print()
    print("[2/3] 检查 Transformers...")
    try:
        import transformers
        print(f"  ✓ Transformers版本: {transformers.__version__}")

        from transformers import pipeline
        print(f"  ✓ Pipeline导入成功")
    except Exception as e:
        print(f"  ✗ Transformers错误: {e}")
        return

    # 3. 测试生成
    print()
    print("[3/3] 测试文本生成 (下载小模型约50MB)...")
    print("  这可能需要几分钟下载模型...")

    try:
        # 使用极小的 GPT2 模型
        generator = pipeline(
            "text-generation",
            model="distilbert/distilgpt2",
            device=-1  # CPU
        )

        result = generator(
            "Hello, how are you today?",
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7
        )

        print(f"  ✓ 生成成功!")
        print(f"  输入: Hello, how are you today?")
        print(f"  输出: {result[0]['generated_text']}")

        print()
        print("=" * 60)
        print("✅ 系统测试通过!")
        print("=" * 60)
        print()
        print("你的环境配置正确，可以继续使用框架。")
        print()
        print("如果需要 GPU 加速:")
        print("  pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu124")

    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        print()
        print("常见问题:")
        print("  1. 网络问题: 确保能访问 huggingface.co")
        print("  2. 磁盘空间: 确保至少有 1GB 空间")
        print("  3. 内存: 确保至少有 4GB 可用内存")


if __name__ == "__main__":
    main()
