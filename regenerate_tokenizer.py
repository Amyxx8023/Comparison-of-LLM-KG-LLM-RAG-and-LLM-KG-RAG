#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成tokenizer文件 - 绕过损坏的tokenizer.json
"""

import os
import sys
import json
import shutil
from datetime import datetime

print("="*60)
print("Tokenizer文件重新生成工具")
print("="*60)
print()

# 切换到项目目录
os.chdir('/home/ypx/mxdmxyy/minimindtrain/MA680/minimind')

# 步骤1: 备份现有tokenizer文件
print("[Step 1] 备份现有tokenizer文件...")
backup_dir = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
try:
    if os.path.exists('model/'):
        shutil.copytree('model/', backup_dir)
        print(f"✓ 已备份到: {backup_dir}")
    else:
        print("✗ model/目录不存在")
        sys.exit(1)
except Exception as e:
    print(f"备份失败: {e}")
    print("继续尝试修复...")

print()

# 步骤2: 删除损坏的tokenizer.json
print("[Step 2] 移除损坏的tokenizer.json...")
try:
    if os.path.exists('model/tokenizer.json'):
        os.rename('model/tokenizer.json', f'model/tokenizer.json.broken')
        print("✓ 已重命名为 tokenizer.json.broken")
    else:
        print("tokenizer.json不存在")
except Exception as e:
    print(f"重命名失败: {e}")

print()

# 步骤3: 尝试加载tokenizer（不使用fast模式）
print("[Step 3] 尝试加载tokenizer（慢速模式）...")
from transformers import AutoTokenizer

try:
    # 不使用fast tokenizer，只依赖tokenizer_config.json
    tokenizer = AutoTokenizer.from_pretrained(
        './model/',
        use_fast=False  # 关键：不使用fast tokenizer
    )
    print("✓ 成功加载慢速tokenizer！")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    
    # 步骤4: 重新保存（不生成tokenizer.json）
    print()
    print("[Step 4] 保存tokenizer到新目录...")
    new_model_dir = 'model_fixed'
    os.makedirs(new_model_dir, exist_ok=True)
    
    # 保存时不生成fast tokenizer的json文件
    tokenizer.save_pretrained(new_model_dir, legacy_format=False)
    
    # 只保留必要的文件
    for file in os.listdir(new_model_dir):
        print(f"  - {file}")
    
    print()
    print("✓ Tokenizer已保存到 model_fixed/")
    print()
    print("="*60)
    print("修复完成！")
    print("="*60)
    print()
    print("下一步:")
    print("1. 测试新tokenizer:")
    print("   python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('./model_fixed/', use_fast=False); print('成功！')\"")
    print()
    print("2. 更新config.py中的TOKENIZER_PATH:")
    print("   TOKENIZER_PATH = './model_fixed/'")
    print()
    print("3. 运行评估:")
    print("   python -m evalmodel.eval_enhanced_models --max_eval_samples 5 --use_simple_rag --no_wandb")
    
except Exception as e:
    print(f"✗ 加载失败: {e}")
    print()
    print("尝试替代方案...")
    print()
    
    # 替代方案：使用通用tokenizer
    print("[Alternative] 使用GPT2 tokenizer作为替代...")
    try:
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # 保存到新目录
        new_model_dir = 'model_gpt2_fallback'
        os.makedirs(new_model_dir, exist_ok=True)
        tokenizer.save_pretrained(new_model_dir)
        
        print(f"✓ GPT2 tokenizer已保存到 {new_model_dir}/")
        print()
        print("⚠️ 注意: 这是一个临时替代方案")
        print("请更新config.py:")
        print(f"   TOKENIZER_PATH = './{new_model_dir}/'")
        
    except Exception as e2:
        print(f"✗ 替代方案也失败: {e2}")
        print()
        print("请联系技术支持或使用原始评估脚本。")

