import torch
import torch.nn as nn

from attention import MultiHeadAttention
from utils import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden, dropout):
        """

        Args:
            d_model : 인코더의 크기
            num_heads : multi-head attention 의 heads 수
            ff_hidden : feed_forward_hidden 일반적으로 d_model * 4
            dropout : drouput
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(num_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, ff_hidden, dropout)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attention = self.attention(
            {
                "query" : x, 
                "key" : x,
                "value" : x,
                "mask" : mask
            }
        )
        
        # droput + residual and norm 1
        attention = self.dropout1(attention)
        attention = self.input_norm(x + attention)
        
        # FF
        output = self.feed_forward(attention)
        
        # dropout + residual and norm 2
        output = self.dropout2(output)
        output = self.output_norm(attention + output)
        
        return output