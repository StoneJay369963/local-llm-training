#!/bin/bash

# 本地LLM训练环境配置脚本
# 适用于 RTX 5060 笔记本电脑 (8GB显存)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "本地LLM训练环境配置"
echo "=========================================="

# 1. 检查Python版本
echo "[1/7] 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 2. 检查CUDA
echo ""
echo "[2/7] 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    nvcc --version 2>/dev/null || echo "nvcc未安装（不影响训练）"
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练"
fi

# 3. 创建虚拟环境
echo ""
echo "[3/7] 创建Python虚拟环境..."
ENV_NAME="llm_env"

if [ -d "$ENV_NAME" ]; then
    echo "虚拟环境已存在: $ENV_NAME"
    read -p "是否重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $ENV_NAME
        python -m venv $ENV_NAME
        echo "已重新创建虚拟环境"
    fi
else
    python -m venv $ENV_NAME
    echo "已创建虚拟环境: $ENV_NAME"
fi

# 4. 激活虚拟环境
echo ""
echo "[4/7] 激活虚拟环境..."
source $ENV_NAME/bin/activate
pip install --upgrade pip
echo "pip版本: $(pip --version)"

# 5. 安装PyTorch
echo ""
echo "[5/7] 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. 安装项目依赖
echo ""
echo "[6/7] 安装项目依赖..."
pip install -r requirements.txt

# 7. 验证安装
echo ""
echo "[7/7] 验证安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 激活环境: source $ENV_NAME/bin/activate"
echo "2. 准备数据: python src/data_processor.py"
echo "3. 开始训练:"
echo "   - 预训练: python scripts/pretrain.py"
echo "   - 微调: python scripts/finetune.py"
echo ""
