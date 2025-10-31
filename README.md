# MiniMind 增强模型评估系统

## 项目简介

本评估系统用于评估MiniMind模型在不同增强策略下的性能表现，包括：
- **LLM+KG**: 知识图谱增强模型
- **LLM+RAG**: 检索增强生成模型
- **LLM+KG+RAG**: 知识图谱+检索增强组合模型

## 评估指标

系统采用五维度评估体系：

1. **ACC (Accuracy)**: 准确性 - 评估回答的事实准确程度
2. **REL (Relevance)**: 相关性 - 评估回答与问题的相关程度
3. **Comp (Completeness)**: 完整性 - 评估回答的完整程度
4. **Clarity**: 清晰度 - 评估回答的表达质量
5. **Overall**: 总体评分 - 四个维度的加权综合评分

所有指标评分范围为 0-10 分。

## 文件结构

```
evalmodel/
├── README.md                      # 本文档
├── REQUIREMENTS.md                # 详细需求文档
├── config.py                      # 配置文件
├── rag_retriever.py              # RAG检索模块
├── metrics_calculator.py         # 指标计算模块
├── eval_enhanced_models.py       # 主评估脚本
├── run_evaluation.sh             # 运行脚本
└── results/                      # 评估结果目录
    ├── enhanced_eval_*.json      # 详细评估结果
    ├── comparison_*.csv          # 模型对比表格
    └── *_predictions.csv         # 各模型预测结果
```

## 环境配置

### 1. 激活Python环境

```bash
conda activate llm
```

### 2. 安装依赖包

```bash
pip install torch transformers wandb openai pandas numpy tqdm rouge nltk scikit-learn
```

### 3. 配置API密钥

API密钥已在 `config.py` 中配置：
- OpenAI API Key: 用于RAG检索和LLM评分
- Wandb API Key: 用于实验记录和可视化

如需修改，请编辑 `config.py` 文件。

## 快速开始

### 方式1: 使用Shell脚本运行（推荐）

```bash
cd /home/ypx/mxdmxyy/minimindtrain/MA680/minimind/evalmodel
bash run_evaluation.sh
```

### 方式2: 直接运行Python脚本

```bash
cd /home/ypx/mxdmxyy/minimindtrain/MA680/minimind
python -m evalmodel.eval_enhanced_models \
    --max_eval_samples 100 \
    --eval_batch_size 20 \
    --use_wandb \
    --device cuda
```

### 方式3: 使用简化版RAG（不需要OpenAI embeddings）

如果OpenAI API调用有问题，可以使用简化版RAG：

```bash
python -m evalmodel.eval_enhanced_models \
    --max_eval_samples 100 \
    --use_simple_rag \
    --use_wandb
```

## 命令行参数

```
--max_eval_samples    评估样本数量（0表示全部，默认100）
--eval_batch_size     每N个样本计算一次批次指标（默认20）
--device              运行设备（cuda/cpu，默认cuda）
--seed                随机种子（默认42）
--use_simple_rag      使用简化版RAG检索器（不依赖OpenAI API）
--use_wandb           启用Wandb日志记录（默认启用）
--no_wandb            禁用Wandb日志记录
```

## 配置说明

### 模型配置 (config.py)

```python
# 待评估的模型
MODELS_TO_EVALUATE = [
    {
        'name': 'LLM+KG',
        'model_path': 'out/full_sft_kg_enhanced_512.pth',
        'use_rag': False,
    },
    {
        'name': 'LLM+RAG',
        'model_path': 'out/full_sft_512.pth',
        'use_rag': True,
    },
    {
        'name': 'LLM+KG+RAG',
        'model_path': 'out/full_sft_kg_enhanced_512.pth',
        'use_rag': True,
    },
]
```

### 评估配置

```python
EVAL_CONFIG = {
    'max_eval_samples': 100,    # 评估样本数
    'eval_batch_size': 20,      # 批次大小
    'show_examples': 5,         # 显示示例数
    'seed': 42,                 # 随机种子
    'device': 'cuda',           # 运行设备
}
```

### RAG配置

```python
RAG_CONFIG = {
    'top_k': 3,                 # 检索文档数量
    'max_context_length': 1500, # 上下文最大长度
}
```

### 指标权重

```python
METRIC_WEIGHTS = {
    'ACC': 0.3,      # 准确性权重
    'REL': 0.25,     # 相关性权重
    'Comp': 0.25,    # 完整性权重
    'Clarity': 0.2,  # 清晰度权重
}
```

## 输出结果

### 1. 终端输出

评估过程中会实时显示：
- 模型加载信息
- 评估进度条
- 前5个示例的详细结果
- 每个批次的平均分数
- 最终的模型对比结果

示例：
```
==================================================
Model Comparison Report
==================================================

    Model  ACC   REL  Comp  Clarity  Overall
  LLM+KG  6.40  6.70  6.60     6.80     8.70
 LLM+RAG  7.20  7.50  7.10     7.30     8.90
LLM+KG+RAG  7.80  8.00  7.60     7.70     9.20
```

### 2. JSON结果文件

位置: `evalmodel/results/enhanced_eval_YYYYMMDD_HHMMSS.json`

包含：
- 每个模型的配置信息
- 最终平均分数
- 前10个样本的预测示例

### 3. CSV对比表格

位置: `evalmodel/results/comparison_YYYYMMDD_HHMMSS.csv`

格式:
```csv
Model,ACC,REL,Comp,Clarity,Overall
LLM+KG,6.40,6.70,6.60,6.80,8.70
LLM+RAG,7.20,7.50,7.10,7.30,8.90
LLM+KG+RAG,7.80,8.00,7.60,7.70,9.20
```

### 4. 详细预测文件

位置: `evalmodel/results/{ModelName}_YYYYMMDD_HHMMSS_predictions.csv`

包含每个样本的：
- Question: 问题
- Reference: 参考答案
- Prediction: 模型预测
- ACC, REL, Comp, Clarity, Overall: 各项分数

### 5. Wandb Dashboard

访问 https://wandb.ai 查看实时评估结果：
- 各模型指标对比图表
- 批次评估趋势
- 示例展示
- 实验参数记录

## 工作流程

1. **初始化**: 加载配置、初始化模型、RAG检索器和指标计算器
2. **数据加载**: 从测试集加载问答对
3. **模型评估**: 对每个模型配置：
   - 加载模型权重
   - 生成预测（使用或不使用RAG）
   - 计算各项指标
   - 记录到Wandb
4. **结果汇总**: 生成模型对比报告
5. **保存输出**: 保存JSON、CSV和日志文件

## RAG检索流程

### 使用OpenAI Embeddings (默认)

1. 加载知识库 (AviationQA.csv)
2. 对查询获取embedding向量
3. 计算与知识库文档的余弦相似度
4. 检索Top-K最相关文档
5. 格式化为上下文注入提示词

### 使用简化版RAG (--use_simple_rag)

1. 加载知识库
2. 使用简单的关键词匹配（Jaccard相似度）
3. 检索Top-K最相关文档
4. 格式化为上下文注入提示词

## 评分机制

### LLM作为评判者（推荐）

使用OpenAI API调用LLM对每个维度进行评分：
- 提供评分标准和示例
- 要求LLM返回0-10的分数
- 低温度设置确保评分稳定性

### 简化评分（Fallback）

当LLM评分不可用时：
- **ACC**: 基于文本相似度（Jaccard）
- **REL**: 基于问题关键词覆盖率
- **Comp**: 基于长度比率和内容覆盖
- **Clarity**: 基于文本特征（长度、重复度等）

## 常见问题

### Q1: OpenAI API调用失败怎么办？

**A**: 使用 `--use_simple_rag` 参数运行，系统会自动切换到简化版RAG和评分。

### Q2: 显存不足怎么办？

**A**: 
- 减少 `--max_eval_samples` 数量
- 使用CPU: `--device cpu`
- 修改 `config.py` 中的 `max_new_tokens`

### Q3: 评估速度太慢怎么办？

**A**:
- 减少评估样本数
- 使用简化版RAG
- 增加 `eval_batch_size` 减少输出频率

### Q4: 如何只评估特定模型？

**A**: 修改 `config.py` 中的 `MODELS_TO_EVALUATE` 列表，注释掉不需要的模型。

### Q5: 如何修改评分权重？

**A**: 修改 `config.py` 中的 `METRIC_WEIGHTS` 字典。

## 扩展开发

### 添加新的评估指标

1. 在 `metrics_calculator.py` 中添加计算函数
2. 在 `calculate_all_scores()` 方法中调用
3. 在报告生成中添加显示逻辑

### 使用自定义知识库

1. 准备CSV或JSONL格式的知识库
2. 修改 `config.py` 中的 `DATASET_PATHS['knowledge_base']`
3. 确保格式兼容（Question/Answer列）

### 添加新的模型配置

在 `config.py` 中添加到 `MODELS_TO_EVALUATE`:

```python
{
    'name': 'YourModelName',
    'model_path': 'path/to/model.pth',
    'use_rag': True/False,
    'description': '模型描述'
}
```

## 性能优化建议

1. **批处理**: 使用较大的 `eval_batch_size` 减少输出开销
2. **缓存**: RAG检索器会缓存embeddings，避免重复计算
3. **并行**: 如果有多张GPU，可以修改代码支持模型并行评估
4. **采样**: 对于大规模评估，先用小样本验证流程

## 联系与支持

如有问题或建议，请联系项目维护者。

## 版本历史

- v1.0 (2025-10-25): 初始版本
  - 支持LLM+KG、LLM+RAG、LLM+KG+RAG评估
  - 五维度评估指标
  - Wandb集成
  - 支持简化版RAG

## 许可证

遵循项目主许可证。

