# LocalLLMTraining Windows 安装脚本
# 使用方法: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LocalLLMTraining Windows 安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 版本
$pythonVersion = python --version 2>&1
Write-Host "检测到 Python: $pythonVersion"

if ($pythonVersion -notmatch "Python 3\.(9|10|11|12)") {
    Write-Host "警告: 推荐使用 Python 3.9-3.12" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[1/4] 安装 PyTorch (CUDA 11.8)..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host ""
Write-Host "[2/4] 安装核心依赖..." -ForegroundColor Green
pip install transformers datasets tokenizers accelerate peft trl
pip install bitsandbytes
pip install fastapi uvicorn[standard] python-multipart jinja2 aiohttp
pip install tenacity
pip install numpy pandas pyarrow
pip install tqdm pyyaml loguru
pip install pydantic pydantic-settings
pip install click inquirer

Write-Host ""
Write-Host "[3/4] 安装可选依赖 (推荐)..." -ForegroundColor Green
pip install tensorboard wandb
pip install sentencepiece tiktoken regex

Write-Host ""
Write-Host "[4/4] 安装量化支持 (可选)..." -ForegroundColor Yellow
Write-Host "提示: GPTQ/GGUF 支持可选，如果不需要可以跳过"
$quantInstall = Read-Host "是否安装 GPTQ 支持? (y/n)"
if ($quantInstall -eq "y") {
    pip install auto-gptq optimum
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "安装完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "快速开始:"
Write-Host "  1. python -m cli.interface init"
Write-Host "  2. python -m cli.interface train"
Write-Host "  3. python -m cli.interface serve"
Write-Host ""
