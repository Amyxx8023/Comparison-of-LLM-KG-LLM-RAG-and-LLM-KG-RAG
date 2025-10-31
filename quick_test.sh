#!/bin/bash
# 快速测试脚本

echo "=========================================="
echo "Quick Tokenizer Test"
echo "=========================================="
echo ""

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Conda环境未激活"
    echo "请运行: conda activate llm"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

cd /home/ypx/mxdmxyy/minimindtrain/MA680/minimind

# 测试1: Python版本和transformers
echo "[Test 1] 检查Python和transformers版本..."
python --version
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
echo ""

# 测试2: 使用默认方式加载tokenizer
echo "[Test 2] 测试tokenizer加载（默认方式）..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    print('✓ 默认方式成功')
    print(f'  Vocab size: {len(tokenizer)}')
except Exception as e:
    print(f'✗ 默认方式失败: {e}')
"
echo ""

# 测试3: 使用slow tokenizer
echo "[Test 3] 测试tokenizer加载（慢速tokenizer）..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('./model/', use_fast=False)
    print('✓ 慢速tokenizer成功')
    print(f'  Vocab size: {len(tokenizer)}')
except Exception as e:
    print(f'✗ 慢速tokenizer失败: {e}')
"
echo ""

# 测试4: 检查tokenizer文件
echo "[Test 4] 检查tokenizer文件..."
if [ -f "model/tokenizer.json" ]; then
    echo "✓ tokenizer.json存在"
    ls -lh model/tokenizer.json
else
    echo "✗ tokenizer.json不存在"
fi

if [ -f "model/tokenizer_config.json" ]; then
    echo "✓ tokenizer_config.json存在"
    ls -lh model/tokenizer_config.json
else
    echo "✗ tokenizer_config.json不存在"
fi
echo ""

# 测试5: 验证原始评估脚本
echo "[Test 5] 测试原始eval_model.py是否能加载tokenizer..."
timeout 10 python -c "
import sys
sys.path.append('.')
from transformers import AutoTokenizer

try:
    # 模拟eval_model.py的加载方式
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    print('✓ 原始脚本的方式可以加载')
except Exception as e:
    print(f'✗ 原始脚本的方式也失败: {e}')
" 2>&1
echo ""

echo "=========================================="
echo "测试完成"
echo "=========================================="
echo ""
echo "如果所有测试都失败，建议："
echo "1. 检查transformers库版本"
echo "2. 运行: python evalmodel/fix_tokenizer.py"
echo "3. 查看: evalmodel/TROUBLESHOOTING.md"

