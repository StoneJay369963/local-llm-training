#!/bin/bash
# LocalLLMTraining 安装脚本 (Linux/Mac)
# 使用方法: bash install.sh

echo "========================================"
echo "LocalLLMTraining 安装脚本"
echo "========================================"
echo ""

# 检查 Python 版本
python_version=$(python --version 2>&1)
echo "检测到 Python: $python_version"

# 检查是否使用 conda
if command -v conda &> /dev/null; then
    echo "检测到 conda 环境"
    read -p "是否在 conda 环境中安装? (y/n) " use_conda
    if [ "$use_conda" = "y" ]; then
        conda create -n llm-training python=3.11
        conda activate llm-training
    fi
fi

echo ""
echo "[1/4] 安装 PyTorch..."
pip install torch torchvision torchaudio

echo ""
echo "[2/4] 安装核心依赖..."
pip install transformers datasets tokenizers accelerate peft trl
pip install bitsandbytes
pip install fastapi uvicorn[standard] python-multipart jinja2 aiohttp
pip install tenacity
pip install numpy pandas pyarrow
pip install tqdm pyyaml loguru
pip install pydantic pydantic-settings
pip install click inquirer

echo ""
echo "[3/4] 安装可选依赖..."
pip install tensorboard wandb
pip install sentencepiece tiktoken regex

echo ""
echo "[4/4] 安装量化支持 (可选)..."
read -p "是否安装量化支持 (GPTQ/GGUF/AWQ)? (y/n) " install_quant
if [ "$install_quant" = "y" ]; then
    pip install auto-gptq optimum
    pip install llama-cpp-python
    pip install autoawq || echo "AWQ 安装失败，可以跳过"
fi

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""
echo "快速开始:"
echo "  python -m cli.interface init"
echo "  python -m cli.interface train"
echo "  python -m cli.interface serve"
echo ""
