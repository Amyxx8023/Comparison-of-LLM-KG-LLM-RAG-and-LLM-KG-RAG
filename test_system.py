#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本 - 验证评估系统各组件是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("MiniMind Enhanced Evaluation System - Component Test")
print("="*60)

# 测试1: 导入模块
print("\n[Test 1] Importing modules...")
try:
    from evalmodel.config import *
    from evalmodel.rag_retriever import RAGRetriever, SimpleRAGRetriever
    from evalmodel.metrics_calculator import MetricsCalculator
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 测试2: 检查配置
print("\n[Test 2] Checking configurations...")
try:
    assert OPENAI_API_KEY, "OpenAI API Key not configured"
    assert OPENAI_BASE_URL, "OpenAI Base URL not configured"
    assert MODEL_PATHS, "Model paths not configured"
    assert DATASET_PATHS, "Dataset paths not configured"
    print("✓ Configurations are valid")
    print(f"  - OpenAI Base URL: {OPENAI_BASE_URL}")
    print(f"  - Models to evaluate: {len(MODELS_TO_EVALUATE)}")
    for model in MODELS_TO_EVALUATE:
        print(f"    * {model['name']}: {model['description']}")
except AssertionError as e:
    print(f"✗ Configuration check failed: {e}")
    sys.exit(1)

# 测试3: 检查文件
print("\n[Test 3] Checking required files...")
files_to_check = [
    ('LLM Model', MODEL_PATHS['llm']),
    ('LLM+KG Model', MODEL_PATHS['llm_kg']),
    ('Test Dataset', DATASET_PATHS['test_data']),
    ('Tokenizer', TOKENIZER_PATH),
]

missing_files = []
for name, path in files_to_check:
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name} not found: {path}")
        missing_files.append(name)

# 知识库是可选的
kb_path = DATASET_PATHS['knowledge_base']
if os.path.exists(kb_path):
    print(f"✓ Knowledge Base: {kb_path}")
else:
    print(f"⚠ Knowledge Base not found (optional): {kb_path}")

if missing_files:
    print(f"\n⚠ Warning: {len(missing_files)} required files are missing")
    print("  The evaluation may not work properly")
else:
    print("\n✓ All required files are present")

# 测试4: 初始化简化版RAG检索器
print("\n[Test 4] Initializing Simple RAG Retriever...")
try:
    config = {
        'DATASET_PATHS': DATASET_PATHS,
        'RAG_CONFIG': RAG_CONFIG,
    }
    retriever = SimpleRAGRetriever(config)
    print(f"✓ Simple RAG Retriever initialized")
    print(f"  - Knowledge base size: {len(retriever.knowledge_base)} documents")
    
    # 测试检索
    test_query = "What is the registration number?"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"  - Test retrieval: {len(results)} results for query '{test_query}'")
    
except Exception as e:
    print(f"✗ RAG Retriever initialization failed: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 初始化指标计算器
print("\n[Test 5] Initializing Metrics Calculator...")
try:
    config = {
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'OPENAI_BASE_URL': OPENAI_BASE_URL,
        'OPENAI_DEFAULT_MODEL': OPENAI_DEFAULT_MODEL,
        'METRIC_WEIGHTS': METRIC_WEIGHTS,
    }
    calculator = MetricsCalculator(config)
    print(f"✓ Metrics Calculator initialized")
    print(f"  - LLM judge available: {calculator.use_llm_judge}")
    
    # 测试计算
    test_prediction = "The registration number is N4YB."
    test_reference = "N4YB"
    test_question = "What is the registration number?"
    
    print("  - Testing simplified scoring...")
    calculator.use_llm_judge = False  # 使用简化评分避免API调用
    scores = calculator.calculate_all_scores(test_prediction, test_reference, test_question)
    print(f"    ACC: {scores.get('ACC', 0):.2f}, "
          f"REL: {scores.get('REL', 0):.2f}, "
          f"Comp: {scores.get('Comp', 0):.2f}, "
          f"Clarity: {scores.get('Clarity', 0):.2f}, "
          f"Overall: {scores.get('Overall', 0):.2f}")
    
except Exception as e:
    print(f"✗ Metrics Calculator initialization failed: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 检查Wandb
print("\n[Test 6] Checking Wandb...")
try:
    import wandb
    print("✓ Wandb package installed")
    if WANDB_API_KEY:
        print(f"✓ Wandb API key configured")
    else:
        print(f"⚠ Wandb API key not configured")
except ImportError:
    print("✗ Wandb package not installed")
    print("  Install with: pip install wandb")

# 测试7: 检查PyTorch和CUDA
print("\n[Test 7] Checking PyTorch and CUDA...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    * GPU {i}: {torch.cuda.get_device_name(i)}")
            # 获取显存信息
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      Memory: {mem_allocated:.2f}GB / {mem_total:.2f}GB")
except Exception as e:
    print(f"✗ PyTorch check failed: {e}")

# 测试8: 检查其他依赖
print("\n[Test 8] Checking other dependencies...")
dependencies = [
    ('transformers', 'Transformers'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('tqdm', 'tqdm'),
    ('openai', 'OpenAI'),
]

missing_deps = []
for module_name, display_name in dependencies:
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
    except ImportError:
        print(f"✗ {display_name} not installed")
        missing_deps.append(display_name)

optional_deps = [
    ('nltk', 'NLTK (for BLEU)'),
    ('rouge', 'Rouge (for ROUGE)'),
    ('sklearn', 'Scikit-learn'),
]

for module_name, display_name in optional_deps:
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
    except ImportError:
        print(f"⚠ {display_name} not installed (optional)")

if missing_deps:
    print(f"\n⚠ Warning: {len(missing_deps)} required dependencies are missing")
    print("  Install with: pip install " + " ".join(missing_deps))

# 总结
print("\n" + "="*60)
print("Test Summary")
print("="*60)

if missing_files:
    print(f"⚠ {len(missing_files)} required files missing")
if missing_deps:
    print(f"⚠ {len(missing_deps)} required dependencies missing")

if not missing_files and not missing_deps:
    print("✓ All tests passed! System is ready for evaluation.")
    print("\nTo run evaluation:")
    print("  cd /home/ypx/mxdmxyy/minimindtrain/MA680/minimind")
    print("  bash evalmodel/run_evaluation.sh")
    print("\nOr:")
    print("  python -m evalmodel.eval_enhanced_models --max_eval_samples 10 --use_simple_rag")
else:
    print("⚠ Some issues detected. Please fix them before running evaluation.")

print("="*60)

