#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - MiniMind Enhanced Model Evaluation
"""

import os

# ==================== OpenAI API 配置 ====================
OPENAI_API_KEY = "sk-kRdNckf5QtXjlsyfJR9h6ciU9tKof6KRwBfmKhIaSYDKi6hd"
OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# ==================== Wandb 配置 ====================
WANDB_API_KEY = "b883c59fe5ad9ccae6eabb7bf66cc7b1dcf1905a"
WANDB_PROJECT = "MiniMind-Enhanced-Evaluation"
WANDB_ENTITY = None  # 如果需要指定组织，在此设置

# ==================== 模型路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    'llm_base': os.path.join(BASE_DIR, 'out/pretrain_512.pth'),
    'llm': os.path.join(BASE_DIR, 'out/full_sft_512.pth'),
    'llm_kg': os.path.join(BASE_DIR, 'out/full_sft_kg_enhanced_512.pth'),
}

# 模型配置
MODEL_CONFIG = {
    'hidden_size': 512,
    'num_hidden_layers': 8,
    'use_moe': False,
}

# ==================== 数据集路径配置 ====================
DATASET_PATHS = {
    'test_data': os.path.join(BASE_DIR, 'dataset/sft_aviationqa_kg.jsonl'),
    'knowledge_base': os.path.join(BASE_DIR, 'dataset/AviationQA.csv'),
    'sft_data': os.path.join(BASE_DIR, 'dataset/sft_aviationqa.jsonl'),
}

# 使用相对路径（与原始脚本保持一致）
TOKENIZER_PATH = './model/'

# ==================== 评估配置 ====================
EVAL_CONFIG = {
    'max_eval_samples': 100,  # 评估样本数量，0表示全部
    'eval_batch_size': 20,     # 每处理N个样本进行一次指标计算
    'show_examples': 5,        # 显示的示例数量
    'seed': 42,                # 随机种子
    'device': 'cuda',          # 'cuda' or 'cpu'
}

# ==================== 生成参数配置 ====================
GENERATION_CONFIG = {
    'max_new_tokens': 512,
    'max_input_length': 1024,
    'temperature': 0.7,
    'top_p': 0.85,
    'do_sample': True,
}

# ==================== RAG 配置 ====================
RAG_CONFIG = {
    'top_k': 3,                    # 检索top-k个文档
    'max_context_length': 1500,    # 检索上下文最大长度
    'chunk_size': 500,             # 文档分块大小
    'chunk_overlap': 50,           # 分块重叠大小
    'use_rerank': False,           # 是否使用重排序
}

# ==================== 评估指标权重 ====================
METRIC_WEIGHTS = {
    'ACC': 0.3,      # 准确性
    'REL': 0.25,     # 相关性
    'Comp': 0.25,    # 完整性
    'Clarity': 0.2,  # 清晰度
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    'results_dir': os.path.join(BASE_DIR, 'evalmodel/results'),
    'save_predictions': True,
    'save_detailed_scores': True,
    'generate_plots': True,
}

# 创建输出目录
os.makedirs(OUTPUT_CONFIG['results_dir'], exist_ok=True)

# ==================== 待评估的模型组合 ====================
MODELS_TO_EVALUATE = [
    {
        'name': 'LLM+KG',
        'model_path': MODEL_PATHS['llm_kg'],
        'use_rag': False,
        'description': '知识图谱增强模型'
    },
    {
        'name': 'LLM+RAG',
        'model_path': MODEL_PATHS['llm'],
        'use_rag': True,
        'description': '基础LLM + RAG检索增强'
    },
    {
        'name': 'LLM+KG+RAG',
        'model_path': MODEL_PATHS['llm_kg'],
        'use_rag': True,
        'description': '知识图谱增强 + RAG检索增强'
    },
]

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(BASE_DIR, 'evalmodel/evaluation.log'),
}

