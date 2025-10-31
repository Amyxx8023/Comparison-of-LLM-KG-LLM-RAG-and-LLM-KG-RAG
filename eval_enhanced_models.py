#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind增强模型评估脚本
评估LLM+KG, LLM+RAG, LLM+KG+RAG三种模型组合
"""

import argparse
import json
import random
import warnings
import numpy as np
import torch
import pandas as pd
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 导入自定义模块
from evalmodel.config import *
from evalmodel.rag_retriever import RAGRetriever, SimpleRAGRetriever
from evalmodel.metrics_calculator import MetricsCalculator

warnings.filterwarnings('ignore')


class EnhancedModelEvaluator:
    """增强模型评估器"""
    
    def __init__(self, args):
        """
        初始化评估器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = args.device
        
        # 合并配置
        self.config = self._merge_config(args)
        
        # 初始化Wandb
        self.wandb = None
        if args.use_wandb:
            self._init_wandb()
        
        # 初始化tokenizer（使用与原始脚本完全相同的方式）
        print(f"Loading tokenizer from: {self.config['TOKENIZER_PATH']}")
        try:
            # 方法1: 完全模仿原始脚本的加载方式
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['TOKENIZER_PATH'])
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"Warning: Default tokenizer loading failed: {e}")
            print("Trying alternative methods...")
            
            # 方法2: 尝试use_fast=False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['TOKENIZER_PATH'],
                    use_fast=False
                )
                print("✓ Tokenizer loaded successfully (slow tokenizer)")
            except Exception as e2:
                # 方法3: 尝试添加trust_remote_code
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config['TOKENIZER_PATH'],
                        use_fast=False,
                        trust_remote_code=True
                    )
                    print("✓ Tokenizer loaded successfully (with trust_remote_code)")
                except Exception as e3:
                    print(f"✗ All tokenizer loading methods failed")
                    print(f"  - Method 1 (default): {str(e)[:100]}")
                    print(f"  - Method 2 (slow): {str(e2)[:100]}")
                    print(f"  - Method 3 (trust): {str(e3)[:100]}")
                    print("\n请运行诊断脚本: python3 evalmodel/fix_tokenizer.py")
                    raise Exception("Failed to load tokenizer. Please check the tokenizer files.")
        
        # 初始化RAG检索器
        if args.use_simple_rag:
            print("使用简化版RAG检索器（不需要OpenAI API）")
            self.rag_retriever = SimpleRAGRetriever(self.config)
        else:
            print("使用OpenAI Embeddings的RAG检索器")
            self.rag_retriever = RAGRetriever(self.config)
        
        # 初始化指标计算器
        self.metrics_calculator = MetricsCalculator(self.config)
        
        # 存储所有模型的评估结果
        self.all_results = {}
    
    def _merge_config(self, args):
        """合并命令行参数和配置文件"""
        config = {
            'OPENAI_API_KEY': OPENAI_API_KEY,
            'OPENAI_BASE_URL': OPENAI_BASE_URL,
            'OPENAI_DEFAULT_MODEL': OPENAI_DEFAULT_MODEL,
            'TOKENIZER_PATH': TOKENIZER_PATH,
            'DATASET_PATHS': DATASET_PATHS,
            'MODEL_CONFIG': MODEL_CONFIG,
            'RAG_CONFIG': RAG_CONFIG,
            'METRIC_WEIGHTS': METRIC_WEIGHTS,
            'OUTPUT_CONFIG': OUTPUT_CONFIG,
        }
        
        # 更新评估配置
        config['EVAL_CONFIG'] = EVAL_CONFIG.copy()
        if args.max_eval_samples > 0:
            config['EVAL_CONFIG']['max_eval_samples'] = args.max_eval_samples
        if args.eval_batch_size > 0:
            config['EVAL_CONFIG']['eval_batch_size'] = args.eval_batch_size
        
        return config
    
    def _init_wandb(self):
        """初始化Wandb"""
        try:
            import wandb
            self.wandb = wandb
            
            # 设置API key
            if WANDB_API_KEY:
                os.environ['WANDB_API_KEY'] = WANDB_API_KEY
            
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"Enhanced-Eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'max_eval_samples': self.config['EVAL_CONFIG']['max_eval_samples'],
                    'eval_batch_size': self.config['EVAL_CONFIG']['eval_batch_size'],
                    'models': [m['name'] for m in MODELS_TO_EVALUATE],
                }
            )
            print("Wandb initialized successfully")
        
        except Exception as e:
            print(f"警告: Wandb初始化失败: {e}")
            self.wandb = None
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        print(f"\nLoading model from: {model_path}")
        
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_hidden_layers=MODEL_CONFIG['num_hidden_layers'],
            use_moe=MODEL_CONFIG['use_moe']
        ))
        
        # 加载权重
        model.load_state_dict(
            torch.load(model_path, map_location=self.device),
            strict=True
        )
        
        model = model.eval().to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
        
        return model
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        data_path = self.config['DATASET_PATHS']['test_data']
        print(f"\nLoading test data from: {data_path}")
        
        test_data = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'conversations' in data:
                        user_msg = data['conversations'][0]['content']
                        assistant_msg = data['conversations'][1]['content']
                        test_data.append({
                            'question': user_msg,
                            'reference': assistant_msg
                        })
        
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
        
        # 随机打乱
        random.seed(self.config['EVAL_CONFIG']['seed'])
        random.shuffle(test_data)
        
        # 限制样本数
        max_samples = self.config['EVAL_CONFIG']['max_eval_samples']
        if max_samples > 0:
            test_data = test_data[:max_samples]
        
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def generate_response(self, model: torch.nn.Module, input_text: str, 
                         use_rag: bool = False) -> str:
        """
        生成模型回复
        
        Args:
            model: 模型
            input_text: 输入文本
            use_rag: 是否使用RAG
            
        Returns:
            生成的回复
        """
        # RAG增强
        if use_rag:
            input_text = self.rag_retriever.augment_prompt(input_text)
        
        # 构建提示词
        messages = [{"role": "user", "content": input_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=GENERATION_CONFIG['max_input_length']
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
                num_return_sequences=1,
                do_sample=GENERATION_CONFIG['do_sample'],
                attention_mask=inputs["attention_mask"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_p=GENERATION_CONFIG['top_p'],
                temperature=GENERATION_CONFIG['temperature']
            )
        
        # 解码
        response = self.tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def evaluate_single_model(self, model_config: Dict, test_data: List[Dict]) -> Dict:
        """
        评估单个模型
        
        Args:
            model_config: 模型配置
            test_data: 测试数据
            
        Returns:
            评估结果
        """
        model_name = model_config['name']
        model_path = model_config['model_path']
        use_rag = model_config['use_rag']
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Description: {model_config['description']}")
        print(f"Use RAG: {use_rag}")
        print(f"{'='*60}")
        
        # 加载模型
        model = self.load_model(model_path)
        
        # 存储结果
        predictions = []
        references = []
        questions = []
        detailed_scores = []
        
        # 批次计数
        batch_num = 0
        batch_predictions = []
        batch_references = []
        batch_questions = []
        
        # 生成预测
        for i, item in enumerate(tqdm(test_data, desc=f"Evaluating {model_name}")):
            question = item['question']
            reference = item['reference']
            
            # 生成预测
            try:
                prediction = self.generate_response(model, question, use_rag=use_rag)
            except Exception as e:
                print(f"\nError generating response for sample {i}: {e}")
                prediction = ""
            
            predictions.append(prediction)
            references.append(reference)
            questions.append(question)
            
            batch_predictions.append(prediction)
            batch_references.append(reference)
            batch_questions.append(question)
            
            # 计算单个样本的详细分数
            try:
                scores = self.metrics_calculator.calculate_all_scores(
                    prediction, reference, question
                )
                detailed_scores.append(scores)
            except Exception as e:
                print(f"\nError calculating scores for sample {i}: {e}")
                detailed_scores.append({})
            
            # 打印示例
            if i < self.config['EVAL_CONFIG']['show_examples']:
                print(f"\n=== Example {i+1} ===")
                print(f"Question: {question[:100]}...")
                print(f"Reference: {reference[:100]}...")
                print(f"Prediction: {prediction[:100]}...")
                if detailed_scores[-1]:
                    print(f"Scores: ACC={detailed_scores[-1].get('ACC', 0):.2f}, "
                          f"REL={detailed_scores[-1].get('REL', 0):.2f}, "
                          f"Comp={detailed_scores[-1].get('Comp', 0):.2f}, "
                          f"Clarity={detailed_scores[-1].get('Clarity', 0):.2f}, "
                          f"Overall={detailed_scores[-1].get('Overall', 0):.2f}")
                print("-" * 50)
            
            # 批次评估
            eval_batch_size = self.config['EVAL_CONFIG']['eval_batch_size']
            if (i + 1) % eval_batch_size == 0 or (i + 1) == len(test_data):
                batch_num += 1
                
                # 计算批次平均分数
                batch_avg_scores = self._calculate_batch_average(detailed_scores[-len(batch_predictions):])
                
                print(f"\n{'='*50}")
                print(f"Batch {batch_num} Results ({model_name}) - Samples: {i+1}")
                print(f"{'='*50}")
                self._print_scores(batch_avg_scores)
                
                # 记录到wandb
                if self.wandb:
                    wandb_data = {
                        f"{model_name}_batch": batch_num,
                        f"{model_name}_samples": i + 1,
                    }
                    for key, value in batch_avg_scores.items():
                        wandb_data[f"{model_name}_{key}"] = value
                    self.wandb.log(wandb_data)
                
                # 清空批次缓存
                batch_predictions = []
                batch_references = []
                batch_questions = []
        
        # 计算最终平均分数
        final_scores = self._calculate_batch_average(detailed_scores)
        
        print(f"\n{'='*60}")
        print(f"Final Results for {model_name}")
        print(f"{'='*60}")
        self._print_scores(final_scores)
        
        # 返回结果
        results = {
            'model_name': model_name,
            'model_config': model_config,
            'predictions': predictions,
            'references': references,
            'questions': questions,
            'detailed_scores': detailed_scores,
            'final_scores': final_scores,
            'num_samples': len(test_data)
        }
        
        # 释放模型内存
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def _calculate_batch_average(self, scores_list: List[Dict]) -> Dict[str, float]:
        """计算批次平均分数"""
        if not scores_list:
            return {}
        
        # 收集所有指标
        all_metrics = {}
        for scores in scores_list:
            for key, value in scores.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # 计算平均值
        avg_scores = {}
        for key, values in all_metrics.items():
            if values:
                avg_scores[key] = np.mean(values)
        
        return avg_scores
    
    def _print_scores(self, scores: Dict[str, float]):
        """打印分数"""
        # 主要指标
        main_metrics = ['ACC', 'REL', 'Comp', 'Clarity', 'Overall']
        for metric in main_metrics:
            if metric in scores:
                print(f"{metric:12s}: {scores[metric]:.4f}")
        
        # 其他指标
        other_metrics = [k for k in scores.keys() if k not in main_metrics]
        if other_metrics:
            print("\nAdditional Metrics:")
            for metric in other_metrics:
                print(f"{metric:12s}: {scores[metric]:.4f}")
    
    def evaluate_all_models(self):
        """评估所有模型"""
        print("\n" + "="*60)
        print("Starting Enhanced Models Evaluation")
        print("="*60)
        
        # 加载测试数据
        test_data = self.load_test_data()
        
        if not test_data:
            print("Error: No test data loaded")
            return
        
        # 评估每个模型
        for model_config in MODELS_TO_EVALUATE:
            try:
                results = self.evaluate_single_model(model_config, test_data)
                self.all_results[model_config['name']] = results
            except Exception as e:
                print(f"\nError evaluating {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成对比报告
        self.generate_comparison_report()
        
        # 保存结果
        self.save_results()
        
        print("\n" + "="*60)
        print("Evaluation Completed!")
        print("="*60)
    
    def generate_comparison_report(self):
        """生成模型对比报告"""
        if not self.all_results:
            return
        
        print("\n" + "="*60)
        print("Model Comparison Report")
        print("="*60)
        
        # 创建对比表格
        comparison_data = []
        for model_name, results in self.all_results.items():
            scores = results['final_scores']
            comparison_data.append({
                'Model': model_name,
                'ACC': scores.get('ACC', 0),
                'REL': scores.get('REL', 0),
                'Comp': scores.get('Comp', 0),
                'Clarity': scores.get('Clarity', 0),
                'Overall': scores.get('Overall', 0),
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
        
        # 记录到wandb
        if self.wandb:
            # 创建对比表格
            table = self.wandb.Table(dataframe=df)
            self.wandb.log({"model_comparison": table})
            
            # 创建对比图表
            for metric in ['ACC', 'REL', 'Comp', 'Clarity', 'Overall']:
                self.wandb.log({
                    f"{metric}_comparison": self.wandb.plot.bar(
                        table, "Model", metric,
                        title=f"{metric} Comparison"
                    )
                })
    
    def save_results(self):
        """保存评估结果"""
        if not self.all_results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config['OUTPUT_CONFIG']['results_dir']
        
        # 保存详细结果
        output_file = os.path.join(results_dir, f"enhanced_eval_{timestamp}.json")
        
        # 准备保存的数据（不包括模型对象）
        save_data = {}
        for model_name, results in self.all_results.items():
            save_data[model_name] = {
                'model_name': results['model_name'],
                'model_config': results['model_config'],
                'final_scores': results['final_scores'],
                'num_samples': results['num_samples'],
                'predictions': results['predictions'][:10],  # 只保存前10个示例
                'references': results['references'][:10],
                'questions': results['questions'][:10],
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # 保存对比表格
        comparison_data = []
        for model_name, results in self.all_results.items():
            scores = results['final_scores']
            comparison_data.append({
                'Model': model_name,
                'ACC': f"{scores.get('ACC', 0):.2f}",
                'REL': f"{scores.get('REL', 0):.2f}",
                'Comp': f"{scores.get('Comp', 0):.2f}",
                'Clarity': f"{scores.get('Clarity', 0):.2f}",
                'Overall': f"{scores.get('Overall', 0):.2f}",
            })
        
        df = pd.DataFrame(comparison_data)
        csv_file = os.path.join(results_dir, f"comparison_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Comparison table saved to: {csv_file}")
        
        # 保存详细的预测结果
        for model_name, results in self.all_results.items():
            predictions_file = os.path.join(
                results_dir,
                f"{model_name.replace('+', '_')}_{timestamp}_predictions.csv"
            )
            
            pred_df = pd.DataFrame({
                'Question': results['questions'],
                'Reference': results['references'],
                'Prediction': results['predictions'],
            })
            
            # 添加分数列
            if results['detailed_scores']:
                for metric in ['ACC', 'REL', 'Comp', 'Clarity', 'Overall']:
                    pred_df[metric] = [s.get(metric, 0) for s in results['detailed_scores']]
            
            pred_df.to_csv(predictions_file, index=False, encoding='utf-8')
            print(f"{model_name} predictions saved to: {predictions_file}")


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="MiniMind Enhanced Models Evaluation"
    )
    
    # 评估参数
    parser.add_argument('--max_eval_samples', default=100, type=int,
                       help='Maximum evaluation samples (0 for all)')
    parser.add_argument('--eval_batch_size', default=20, type=int,
                       help='Evaluate every N samples')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       type=str, help='Device to use')
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed')
    
    # RAG参数
    parser.add_argument('--use_simple_rag', action='store_true',
                       help='Use simple RAG retriever instead of OpenAI embeddings')
    
    # Wandb参数
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='Disable wandb logging')
    parser.set_defaults(use_wandb=True)
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建评估器并运行
    try:
        evaluator = EnhancedModelEvaluator(args)
        evaluator.evaluate_all_models()
        
        # 关闭wandb
        if evaluator.wandb:
            evaluator.wandb.finish()
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

