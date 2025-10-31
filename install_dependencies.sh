#!/bin/bash
# 安装评估系统所需的依赖包

echo "=========================================="
echo "Installing Dependencies for MiniMind Enhanced Evaluation System"
echo "=========================================="
echo ""

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "错误: Conda环境未激活"
    echo "请先运行: conda activate llm"
    exit 1
fi

echo "当前Conda环境: $CONDA_DEFAULT_ENV"
echo ""

# 必需的依赖包
echo "安装必需的依赖包..."
pip install -q pandas numpy tqdm

# PyTorch (如果未安装)
echo "检查PyTorch..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装PyTorch..."
    pip install torch torchvision torchaudio
else
    echo "PyTorch已安装"
fi

# Transformers
echo "检查Transformers..."
python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装Transformers..."
    pip install transformers
else
    echo "Transformers已安装"
fi

# OpenAI
echo "安装OpenAI SDK..."
pip install -q openai

# Wandb
echo "安装Wandb..."
pip install -q wandb

# 可选的依赖包
echo ""
echo "安装可选的依赖包..."
pip install -q nltk rouge scikit-learn

# 下载NLTK数据
echo ""
echo "下载NLTK数据..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"

echo ""
echo "=========================================="
echo "依赖安装完成！"
echo "=========================================="
echo ""
echo "运行测试验证:"
echo "  python3 evalmodel/test_system.py"
echo ""

