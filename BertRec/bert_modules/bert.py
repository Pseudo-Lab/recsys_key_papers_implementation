import torch
import torch.nn as nn

from embedding import BERTembedding
from encoder import EncoderBlock

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        max_len = args.max_len
        num_items = args.num_items
        n_layers = args.num_blocks
        num_heads = args.num_heads
        vocab_size = num_items + 2
        hidden = args.hidden_unit
        dropout = args.dropout
        
        self.hidden = hidden

        self.embedding = BERTembedding(vocab_size, hidden, max_len, dropout)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden, num_heads, hidden * 4, dropout) for _ in range(n_layers)
        ])
        
        self.out = nn.Linear(hidden, num_items + 1)
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        x = self.embedding(x)
        
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        
        output = self.out(x)
        
        return output