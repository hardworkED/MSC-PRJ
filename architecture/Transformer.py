import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.dim_k = dim // n_heads
        
        self.weight_Q = nn.Linear(dim, dim)
        self.weight_K = nn.Linear(dim, dim)
        self.weight_V = nn.Linear(dim, dim)
        self.weight_out = nn.Linear(dim, dim)

    def scaled_dot_product_attention(self, Q, K, V):
        scaled_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_k)
        p = torch.softmax(scaled_scores, dim=-1)
        output = torch.matmul(p, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.dim_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim)
        
    @autocast()
    def forward(self, Q, K, V):
        Q = self.split_heads(self.weight_Q(Q))
        K = self.split_heads(self.weight_K(K))
        V = self.split_heads(self.weight_V(V))
        
        out = self.scaled_dot_product_attention(Q, K, V)
        out = self.weight_out(self.combine_heads(out))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        encode = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) * (torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)))
        encode[:, 0::2] = torch.sin(pos)
        encode[:, 1::2] = torch.cos(pos)

        self.encode = encode.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encode[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, dim=512, n_heads=16, dropout=0.1):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(dim, n_heads)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    @autocast()
    def forward(self, x):
        out = self.MultiHeadAttention(x, x, x)
        x = self.norm1(x + self.dropout(out))
        out = self.PositionWiseFeedForward(x)
        x = self.norm2(x + self.dropout(out))
        return x

class TransformerEncoderOnly(nn.Module):
    def __init__(self, dim, n_heads=16, dropout=0.1, n_encoders=6):
        super(TransformerEncoderOnly, self).__init__()
        self.emb = nn.Embedding(1000, dim)
        self.dropout = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([Encoder(dim, n_heads, dropout) for _ in range(n_encoders)])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    @autocast()
    def forward(self, x):
        B, T, D = x.shape
        out = self.dropout(x + self.emb.weight[:x.shape[1]])
        for encoder in self.encoders:
            out = encoder(out)

        out += x
        out = out.permute(0, 2, 1)
        out = self.avg_pool(out)
        out = out.reshape(B, -1)
        return out