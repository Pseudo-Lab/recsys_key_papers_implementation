import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Attention(nn.Module):
    """
    scaled dot product attention 계산
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.tranpose(-2, 1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    모델 크기와 헤드의 수를 입력을 받음
    """
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 모델의 차원에서 배치 단위로 모든 선형 투영을 수행 => num_heads x d_k
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 어텐션 적용
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)
        
        # concat 후 output lienear 적용
        output = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.output_linear(output)
        
        return output