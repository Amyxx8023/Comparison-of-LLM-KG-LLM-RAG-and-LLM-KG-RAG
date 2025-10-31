# 快速开始指南

## 1. 激活Conda环境

```bash
conda activate llm
```

## 2. 安装依赖

```bash
cd /home/ypx/mxdmxyy/minimindtrain/MA680/minimind
chmod +x evalmodel/install_dependencies.sh
bash evalmodel/install_dependencies.sh
```

或手动安装：

```bash
pip install torch transformers pandas numpy tqdm openai wandb nltk rouge scikit-learn
python3 -c "import nltk; nltk.download('punkt')"
```

## 3. 运行系统测试

```bash
python3 evalmodel/test_system.py
```

这将检查：
- 所有模块是否正确导入
- 配置是否有效
- 必要文件是否存在
- RAG检索器是否工作
- 指标计算器是否工作
- 依赖包是否安装

## 4. 开始评估

### 方式A: 使用Shell脚本（推荐）

```bash
# 默认评估100个样本
bash evalmodel/run_evaluation.sh

# 评估10个样本（快速测试）
bash evalmodel/run_evaluation.sh 10 5 cuda

# 使用简化版RAG（不需要OpenAI embeddings API）
bash evalmodel/run_evaluation.sh 10 5 cuda true
```

### 方式B: 直接运行Python脚本

```bash
# 完整评估（使用OpenAI embeddings）
python3 -m evalmodel.eval_enhanced_models \
    --max_eval_samples 100 \
    --eval_batch_size 20 \
    --use_wandb \
    --device cuda

# 快速测试（使用简化RAG）
python3 -m evalmodel.eval_enhanced_models \
    --max_eval_samples 10 \
    --eval_batch_size 5 \
    --use_simple_rag \
    --use_wandb \
    --device cuda
```

### 方式C: 不使用Wandb

```bash
python3 -m evalmodel.eval_enhanced_models \
    --max_eval_samples 10 \
    --use_simple_rag \
    --no_wandb
```

## 5. 查看结果

评估完成后，结果将保存在 `evalmodel/results/` 目录：

```bash
# 查看对比表格
cat evalmodel/results/comparison_*.csv

# 查看详细结果
ls -lh evalmodel/results/
```

## 常见问题

### Q1: ModuleNotFoundError

**A**: 确保已激活conda环境并安装了所有依赖：
```bash
conda activate llm
bash evalmodel/install_dependencies.sh
```

### Q2: CUDA out of memory

**A**: 减少评估样本数或使用CPU：
```bash
python3 -m evalmodel.eval_enhanced_models \
    --max_eval_samples 10 \
    --device cpu \
    --use_simple_rag
```

### Q3: OpenAI API错误

**A**: 使用简化版RAG：
```bash
python3 -m evalmodel.eval_enhanced_models \
    --use_simple_rag \
    --max_eval_samples 10
```

### Q4: 模型文件不存在

**A**: 检查模型路径是否正确：
```bash
ls -lh out/*.pth
```

如果路径不同，修改 `evalmodel/config.py` 中的 `MODEL_PATHS`。

## 预期输出

### 终端输出示例

```
============================================================
Starting Enhanced Models Evaluation
============================================================

Loading test data from: dataset/sft_aviationqa_kg.jsonl
Loaded 100 test samples

============================================================
Evaluating: LLM+KG
Description: 知识图谱增强模型
Use RAG: False
============================================================

Loading model from: out/full_sft_kg_enhanced_512.pth
Model parameters: 26.00M

Evaluating LLM+KG: 100%|██████████| 100/100 [10:00<00:00, 6.00s/it]

==================================================
Batch 1 Results (LLM+KG) - Samples: 20
==================================================
ACC         : 6.4500
REL         : 6.7200
Comp        : 6.6100
Clarity     : 6.8300
Overall     : 8.7100

...

============================================================
Model Comparison Report
============================================================

       Model   ACC   REL  Comp  Clarity  Overall
     LLM+KG  6.40  6.70  6.60     6.80     8.70
    LLM+RAG  7.20  7.50  7.10     7.30     8.90
LLM+KG+RAG  7.80  8.00  7.60     7.70     9.20

Results saved to: evalmodel/results/enhanced_eval_20251025_120000.json
Comparison table saved to: evalmodel/results/comparison_20251025_120000.csv

============================================================
Evaluation Completed!
============================================================
```

### 文件输出

- `enhanced_eval_*.json` - 详细评估结果
- `comparison_*.csv` - 模型对比表格
- `LLM_KG_*_predictions.csv` - LLM+KG模型的预测
- `LLM_RAG_*_predictions.csv` - LLM+RAG模型的预测
- `LLM_KG_RAG_*_predictions.csv` - LLM+KG+RAG模型的预测

## 下一步

1. 查看Wandb Dashboard: https://wandb.ai/
2. 分析CSV结果文件
3. 根据需要调整配置（`evalmodel/config.py`）
4. 重新运行评估

## 技术支持

如遇问题，请：
1. 查看 `evalmodel/README.md` 详细文档
2. 运行 `python3 evalmodel/test_system.py` 诊断
3. 检查 `evalmodel/evaluation.log` 日志文件

