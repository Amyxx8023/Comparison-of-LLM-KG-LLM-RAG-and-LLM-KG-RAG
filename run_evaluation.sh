#!/bin/bash
# MiniMind增强模型评估运行脚本

echo "=========================================="
echo "MiniMind Enhanced Models Evaluation"
echo "=========================================="
echo ""

# 设置环境
export PYTHONPATH="${PYTHONPATH}:/home/ypx/mxdmxyy/minimindtrain/MA680/minimind"

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project Root: $PROJECT_ROOT"
echo "Script Directory: $SCRIPT_DIR"
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: Conda环境未激活"
    echo "请运行: conda activate llm"
    exit 1
fi

echo "当前Conda环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查必要的文件
echo "检查必要文件..."

# 检查模型文件
if [ ! -f "out/full_sft_512.pth" ]; then
    echo "错误: 模型文件 out/full_sft_512.pth 不存在"
    exit 1
fi

if [ ! -f "out/full_sft_kg_enhanced_512.pth" ]; then
    echo "错误: 模型文件 out/full_sft_kg_enhanced_512.pth 不存在"
    exit 1
fi

# 检查数据集
if [ ! -f "dataset/sft_aviationqa_kg.jsonl" ]; then
    echo "错误: 数据集文件 dataset/sft_aviationqa_kg.jsonl 不存在"
    exit 1
fi

if [ ! -f "dataset/AviationQA.csv" ]; then
    echo "警告: 知识库文件 dataset/AviationQA.csv 不存在，RAG功能可能无法正常工作"
fi

echo "文件检查完成"
echo ""

# 设置默认参数
MAX_SAMPLES=${1:-100}
BATCH_SIZE=${2:-20}
DEVICE=${3:-cuda}
USE_SIMPLE_RAG=${4:-false}

echo "评估参数:"
echo "  - 评估样本数: $MAX_SAMPLES"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 运行设备: $DEVICE"
echo "  - 使用简化RAG: $USE_SIMPLE_RAG"
echo ""

# 构建命令
CMD="python -m evalmodel.eval_enhanced_models \
    --max_eval_samples $MAX_SAMPLES \
    --eval_batch_size $BATCH_SIZE \
    --device $DEVICE \
    --use_wandb"

# 添加简化RAG选项
if [ "$USE_SIMPLE_RAG" = "true" ]; then
    CMD="$CMD --use_simple_rag"
    echo "注意: 使用简化版RAG（不依赖OpenAI embeddings API）"
    echo ""
fi

echo "开始评估..."
echo "命令: $CMD"
echo ""

# 运行评估
eval $CMD

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "评估完成！"
    echo "=========================================="
    echo ""
    echo "结果文件保存在: $PROJECT_ROOT/evalmodel/results/"
    echo ""
    echo "查看结果:"
    echo "  - JSON详细结果: evalmodel/results/enhanced_eval_*.json"
    echo "  - CSV对比表格: evalmodel/results/comparison_*.csv"
    echo "  - 各模型预测: evalmodel/results/*_predictions.csv"
    echo ""
    echo "Wandb Dashboard: https://wandb.ai/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "评估过程出现错误"
    echo "=========================================="
    echo ""
    echo "请检查错误信息并重试"
    echo ""
    exit 1
fi

