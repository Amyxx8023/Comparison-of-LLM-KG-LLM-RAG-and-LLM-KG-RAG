#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索模块 - 基于OpenAI API的检索增强生成
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
import warnings

warnings.filterwarnings('ignore')


class RAGRetriever:
    """RAG检索器 - 使用OpenAI embeddings进行向量检索"""
    
    def __init__(self, config):
        """
        初始化RAG检索器
        
        Args:
            config: 配置字典，包含API密钥、知识库路径等
        """
        self.config = config
        self.api_key = config.get('OPENAI_API_KEY')
        self.base_url = config.get('OPENAI_BASE_URL')
        self.model = config.get('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini')
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # RAG配置
        self.top_k = config.get('RAG_CONFIG', {}).get('top_k', 3)
        self.max_context_length = config.get('RAG_CONFIG', {}).get('max_context_length', 1500)
        
        # 加载知识库
        self.knowledge_base = []
        self.embeddings_cache = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """加载知识库"""
        kb_path = self.config.get('DATASET_PATHS', {}).get('knowledge_base')
        
        if not kb_path or not os.path.exists(kb_path):
            print(f"警告: 知识库文件不存在: {kb_path}")
            return
        
        print(f"Loading knowledge base from: {kb_path}")
        
        try:
            if kb_path.endswith('.csv'):
                df = pd.read_csv(kb_path)
                # 假设CSV包含Question和Answer列
                for idx, row in df.iterrows():
                    question = str(row.get('Question', ''))
                    answer = str(row.get('Answer', ''))
                    if question and answer:
                        self.knowledge_base.append({
                            'id': idx,
                            'question': question,
                            'answer': answer,
                            'text': f"Question: {question}\nAnswer: {answer}"
                        })
            elif kb_path.endswith('.jsonl'):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        data = json.loads(line)
                        if 'conversations' in data:
                            question = data['conversations'][0]['content']
                            answer = data['conversations'][1]['content']
                            self.knowledge_base.append({
                                'id': idx,
                                'question': question,
                                'answer': answer,
                                'text': f"Question: {question}\nAnswer: {answer}"
                            })
            
            print(f"Loaded {len(self.knowledge_base)} documents into knowledge base")
        
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量
        """
        # 检查缓存
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            # 调用OpenAI API获取embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # 使用小模型降低成本
                input=text
            )
            embedding = response.data[0].embedding
            
            # 缓存结果
            self.embeddings_cache[text] = embedding
            return embedding
        
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # 返回零向量作为fallback
            return [0.0] * 1536  # text-embedding-3-small的维度
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
            
        Returns:
            检索到的文档列表
        """
        if not self.knowledge_base:
            print("警告: 知识库为空")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # 获取查询的embedding
        query_embedding = self.get_embedding(query)
        
        # 计算与所有文档的相似度
        similarities = []
        for doc in self.knowledge_base:
            doc_embedding = self.get_embedding(doc['text'])
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                'doc': doc,
                'similarity': similarity
            })
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]
        
        return [item['doc'] for item in top_results]
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        将检索到的文档格式化为上下文
        
        Args:
            retrieved_docs: 检索到的文档列表
            
        Returns:
            格式化的上下文字符串
        """
        if not retrieved_docs:
            return ""
        
        context_parts = ["以下是相关的参考信息：\n"]
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n参考 {i}:")
            context_parts.append(f"问题: {doc['question']}")
            context_parts.append(f"答案: {doc['answer']}\n")
        
        context = "\n".join(context_parts)
        
        # 限制上下文长度
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    def augment_prompt(self, query: str) -> str:
        """
        使用RAG增强提示词
        
        Args:
            query: 原始查询
            
        Returns:
            增强后的提示词
        """
        # 检索相关文档
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return query
        
        # 格式化上下文
        context = self.format_context(retrieved_docs)
        
        # 构建增强提示词
        augmented_prompt = f"""{context}

请基于以上参考信息回答下面的问题。如果参考信息中没有相关内容，请根据你的知识回答。

问题: {query}"""
        
        return augmented_prompt
    
    def retrieve_with_llm_rerank(self, query: str, top_k: int = None) -> List[Dict]:
        """
        使用LLM对检索结果进行重排序（可选功能）
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
            
        Returns:
            重排序后的文档列表
        """
        # 先进行常规检索，获取更多候选
        candidates = self.retrieve(query, top_k=(top_k or self.top_k) * 2)
        
        if not candidates:
            return []
        
        # TODO: 使用LLM评估每个候选的相关性
        # 这里暂时返回原始检索结果
        return candidates[:top_k or self.top_k]


class SimpleRAGRetriever:
    """简化版RAG检索器 - 不依赖外部API，使用简单的BM25或TF-IDF"""
    
    def __init__(self, config):
        """初始化简化版检索器"""
        self.config = config
        self.top_k = config.get('RAG_CONFIG', {}).get('top_k', 3)
        self.max_context_length = config.get('RAG_CONFIG', {}).get('max_context_length', 1500)
        
        self.knowledge_base = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """加载知识库"""
        kb_path = self.config.get('DATASET_PATHS', {}).get('knowledge_base')
        
        if not kb_path or not os.path.exists(kb_path):
            print(f"警告: 知识库文件不存在: {kb_path}")
            return
        
        print(f"Loading knowledge base from: {kb_path}")
        
        try:
            if kb_path.endswith('.csv'):
                df = pd.read_csv(kb_path)
                for idx, row in df.iterrows():
                    question = str(row.get('Question', ''))
                    answer = str(row.get('Answer', ''))
                    if question and answer:
                        self.knowledge_base.append({
                            'id': idx,
                            'question': question,
                            'answer': answer,
                            'text': f"Question: {question}\nAnswer: {answer}"
                        })
            
            print(f"Loaded {len(self.knowledge_base)} documents")
        
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    
    def simple_score(self, query: str, doc_text: str) -> float:
        """简单的关键词匹配评分"""
        query_words = set(query.lower().split())
        doc_words = set(doc_text.lower().split())
        
        if not query_words:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """简单检索"""
        if not self.knowledge_base:
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # 计算每个文档的得分
        scored_docs = []
        for doc in self.knowledge_base:
            score = self.simple_score(query, doc['text'])
            scored_docs.append({'doc': doc, 'score': score})
        
        # 排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return [item['doc'] for item in scored_docs[:top_k]]
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """格式化上下文"""
        if not retrieved_docs:
            return ""
        
        context_parts = ["以下是相关的参考信息：\n"]
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n参考 {i}:")
            context_parts.append(f"问题: {doc['question']}")
            context_parts.append(f"答案: {doc['answer']}\n")
        
        context = "\n".join(context_parts)
        
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    def augment_prompt(self, query: str) -> str:
        """增强提示词"""
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return query
        
        context = self.format_context(retrieved_docs)
        
        augmented_prompt = f"""{context}

请基于以上参考信息回答下面的问题。如果参考信息中没有相关内容，请根据你的知识回答。

问题: {query}"""
        
        return augmented_prompt

