#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标计算模块 - 计算ACC, REL, Comp, Clarity, Overall等评估指标
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
import warnings
import time

warnings.filterwarnings('ignore')

# 尝试导入NLTK和ROUGE
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, config):
        """
        初始化指标计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.api_key = config.get('OPENAI_API_KEY')
        self.base_url = config.get('OPENAI_BASE_URL')
        self.model = config.get('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini')
        
        # 初始化OpenAI客户端（用于LLM评分）
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.use_llm_judge = True
        except Exception as e:
            print(f"警告: 无法初始化OpenAI客户端，将使用简化评分: {e}")
            self.use_llm_judge = False
        
        # 初始化ROUGE
        self.rouge = Rouge() if HAS_ROUGE else None
        self.smoothing = SmoothingFunction().method1 if HAS_NLTK else None
    
    def calculate_acc_score(self, prediction: str, reference: str) -> float:
        """
        计算准确性分数 (ACC - Accuracy)
        使用LLM判断或基于文本相似度的简化方法
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            
        Returns:
            0-10的分数
        """
        if self.use_llm_judge:
            try:
                return self._llm_judge_score(prediction, reference, 'accuracy')
            except Exception as e:
                print(f"LLM评分失败，使用简化方法: {e}")
        
        # 简化方法：基于文本匹配
        pred_clean = self._clean_text(prediction)
        ref_clean = self._clean_text(reference)
        
        # 计算词级别的重叠
        pred_words = set(pred_clean.split())
        ref_words = set(ref_clean.split())
        
        if not ref_words:
            return 0.0
        
        # Jaccard相似度
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 转换为0-10分
        return jaccard * 10.0
    
    def calculate_rel_score(self, prediction: str, reference: str, question: str) -> float:
        """
        计算相关性分数 (REL - Relevance)
        评估回答与问题的相关程度
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            question: 问题
            
        Returns:
            0-10的分数
        """
        if self.use_llm_judge:
            try:
                return self._llm_judge_score(prediction, reference, 'relevance', question)
            except Exception as e:
                print(f"LLM评分失败，使用简化方法: {e}")
        
        # 简化方法：问题关键词在答案中的覆盖率
        question_words = set(self._clean_text(question).split())
        pred_words = set(self._clean_text(prediction).split())
        
        if not question_words:
            return 5.0  # 默认分数
        
        # 计算问题关键词覆盖率
        coverage = len(question_words.intersection(pred_words)) / len(question_words)
        
        # 转换为0-10分
        return coverage * 10.0
    
    def calculate_comp_score(self, prediction: str, reference: str) -> float:
        """
        计算完整性分数 (Comp - Completeness)
        评估回答的完整程度
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            
        Returns:
            0-10的分数
        """
        if self.use_llm_judge:
            try:
                return self._llm_judge_score(prediction, reference, 'completeness')
            except Exception as e:
                print(f"LLM评分失败，使用简化方法: {e}")
        
        # 简化方法：基于长度比率和内容覆盖
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        
        # 长度比率
        length_ratio = min(pred_len, ref_len) / max(pred_len, ref_len, 1)
        
        # 内容覆盖率
        ref_words = set(self._clean_text(reference).split())
        pred_words = set(self._clean_text(prediction).split())
        
        coverage = len(ref_words.intersection(pred_words)) / len(ref_words) if ref_words else 0.0
        
        # 综合评分
        score = (length_ratio * 0.3 + coverage * 0.7) * 10.0
        
        return score
    
    def calculate_clarity_score(self, prediction: str) -> float:
        """
        计算清晰度分数 (Clarity)
        评估回答的表达清晰度
        
        Args:
            prediction: 模型预测
            
        Returns:
            0-10的分数
        """
        if self.use_llm_judge:
            try:
                return self._llm_judge_clarity(prediction)
            except Exception as e:
                print(f"LLM评分失败，使用简化方法: {e}")
        
        # 简化方法：基于文本特征
        score = 10.0
        
        # 长度检查
        word_count = len(prediction.split())
        if word_count < 5:
            score -= 3  # 太短
        elif word_count > 200:
            score -= 2  # 太长可能啰嗦
        
        # 重复词检查
        words = prediction.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 2  # 重复度高
        
        # 标点符号检查
        if not any(p in prediction for p in ['.', '!', '?', '。', '！', '？']):
            score -= 1  # 缺少标点
        
        return max(0.0, score)
    
    def calculate_overall_score(self, acc: float, rel: float, comp: float, clarity: float) -> float:
        """
        计算总体分数 (Overall)
        基于各项指标的加权平均
        
        Args:
            acc: 准确性分数
            rel: 相关性分数
            comp: 完整性分数
            clarity: 清晰度分数
            
        Returns:
            0-10的分数
        """
        weights = self.config.get('METRIC_WEIGHTS', {
            'ACC': 0.3,
            'REL': 0.25,
            'Comp': 0.25,
            'Clarity': 0.2
        })
        
        overall = (
            acc * weights['ACC'] +
            rel * weights['REL'] +
            comp * weights['Comp'] +
            clarity * weights['Clarity']
        )
        
        return overall
    
    def calculate_all_scores(self, prediction: str, reference: str, question: str) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            question: 问题
            
        Returns:
            包含所有指标的字典
        """
        scores = {}
        
        # 计算各项指标
        scores['ACC'] = self.calculate_acc_score(prediction, reference)
        scores['REL'] = self.calculate_rel_score(prediction, reference, question)
        scores['Comp'] = self.calculate_comp_score(prediction, reference)
        scores['Clarity'] = self.calculate_clarity_score(prediction)
        
        # 计算总体分数
        scores['Overall'] = self.calculate_overall_score(
            scores['ACC'], scores['REL'], scores['Comp'], scores['Clarity']
        )
        
        # 额外的自动指标
        if HAS_NLTK:
            scores['BLEU'] = self._calculate_bleu(prediction, reference)
        
        if HAS_ROUGE:
            rouge_scores = self._calculate_rouge(prediction, reference)
            scores.update(rouge_scores)
        
        return scores
    
    def _llm_judge_score(self, prediction: str, reference: str, aspect: str, question: str = None) -> float:
        """
        使用LLM作为评判者评分
        
        Args:
            prediction: 模型预测
            reference: 参考答案
            aspect: 评估方面 ('accuracy', 'relevance', 'completeness')
            question: 问题（用于relevance评估）
            
        Returns:
            0-10的分数
        """
        # 构建评估提示词
        if aspect == 'accuracy':
            prompt = f"""请评估以下答案的准确性。

参考答案：{reference}

待评估答案：{prediction}

请仅从事实准确性角度评分（0-10分）：
- 10分：完全准确，所有事实正确
- 7-9分：大部分准确，有少量小错误
- 4-6分：部分准确，有明显错误
- 1-3分：大部分错误
- 0分：完全错误或无关

请只回复一个0-10之间的数字分数。"""

        elif aspect == 'relevance':
            prompt = f"""请评估以下答案与问题的相关性。

问题：{question}

参考答案：{reference}

待评估答案：{prediction}

请评估答案与问题的相关程度（0-10分）：
- 10分：直接回答问题核心，高度相关
- 7-9分：基本相关，略有偏离
- 4-6分：部分相关，有一定偏离
- 1-3分：相关性低
- 0分：完全不相关

请只回复一个0-10之间的数字分数。"""

        elif aspect == 'completeness':
            prompt = f"""请评估以下答案的完整性。

参考答案：{reference}

待评估答案：{prediction}

请评估答案的完整程度（0-10分）：
- 10分：全面完整，涵盖所有要点
- 7-9分：较完整，缺少次要信息
- 4-6分：部分完整，缺少重要信息
- 1-3分：不够完整
- 0分：严重不完整或无内容

请只回复一个0-10之间的数字分数。"""
        
        else:
            return 5.0
        
        try:
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的评估专家，负责客观公正地评估答案质量。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度以获得稳定评分
                max_tokens=10
            )
            
            # 解析分数
            score_text = response.choices[0].message.content.strip()
            # 提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', score_text)
            if numbers:
                score = float(numbers[0])
                return min(10.0, max(0.0, score))  # 限制在0-10
            else:
                return 5.0  # 默认分数
        
        except Exception as e:
            print(f"LLM评分出错: {e}")
            return 5.0
    
    def _llm_judge_clarity(self, prediction: str) -> float:
        """使用LLM评估清晰度"""
        prompt = f"""请评估以下答案的清晰度和表达质量。

答案：{prediction}

请评估答案的清晰度（0-10分）：
- 10分：表达清晰，逻辑连贯，易于理解
- 7-9分：基本清晰，略有模糊
- 4-6分：部分清晰，有理解困难
- 1-3分：不够清晰，难以理解
- 0分：混乱不清

请只回复一个0-10之间的数字分数。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            import re
            numbers = re.findall(r'\d+\.?\d*', score_text)
            if numbers:
                score = float(numbers[0])
                return min(10.0, max(0.0, score))
            else:
                return 5.0
        
        except Exception as e:
            print(f"LLM评分出错: {e}")
            return 5.0
    
    def _calculate_bleu(self, prediction: str, reference: str) -> float:
        """计算BLEU分数"""
        if not HAS_NLTK:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]
        
        if not pred_tokens:
            return 0.0
        
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing)
            return score
        except:
            return 0.0
    
    def _calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not HAS_ROUGE or not prediction.strip() or not reference.strip():
            return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
        
        try:
            scores = self.rouge.get_scores(prediction, reference)[0]
            return {
                'ROUGE-1': scores['rouge-1']['f'],
                'ROUGE-2': scores['rouge-2']['f'],
                'ROUGE-L': scores['rouge-l']['f']
            }
        except:
            return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 转小写
        text = text.lower()
        # 移除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def batch_calculate(self, predictions: List[str], references: List[str], 
                       questions: List[str]) -> Dict[str, List[float]]:
        """
        批量计算指标
        
        Args:
            predictions: 预测列表
            references: 参考答案列表
            questions: 问题列表
            
        Returns:
            包含各指标列表的字典
        """
        batch_scores = {
            'ACC': [],
            'REL': [],
            'Comp': [],
            'Clarity': [],
            'Overall': []
        }
        
        for pred, ref, question in zip(predictions, references, questions):
            scores = self.calculate_all_scores(pred, ref, question)
            for key in batch_scores.keys():
                batch_scores[key].append(scores[key])
        
        return batch_scores
    
    def aggregate_scores(self, batch_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """
        聚合批次分数
        
        Args:
            batch_scores: 批次分数字典
            
        Returns:
            平均分数字典
        """
        aggregated = {}
        for key, values in batch_scores.items():
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated

