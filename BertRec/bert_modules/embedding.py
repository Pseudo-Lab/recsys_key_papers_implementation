from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model) -> None:
        super().__init__()
        
        # 학습이 가능한 형태의 positional embedding
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size=512):
        super().__init__(vocab_size, embedding_size, padding_idx=0)

class BERTembedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, dropout=0.1):
        super().__init__()
        
        self.token = TokenEmbedding(vocab_size, embedding_size)
        self.position = PositionalEmbedding(max_len, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        
    def forward(self, sequence):
        output = self.token(sequence) + self.position(sequence)
        output = self.dropout(output)
        
        return output