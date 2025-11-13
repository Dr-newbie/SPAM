# -*- coding: utf-8 -*-
# code/utils/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Q(from A) x K/V(from B) → attended A
    모든 입력은 (B, N, D). D가 다르면 입력 투영으로 정렬.
    """
    def __init__(self, dim_q: int, dim_kv: int, dim_out: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        # Q,K,V: (B,N,Dq/Dkv)
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)
        x, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = self.norm(x + q)  # residual to q
        return x

class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int = 4_096, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return self.norm(x + h)

class CrossAttentionBlock(nn.Module):
    """
    (1) CrossAttn + (2) FFN
    """
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=8, dropout=0.1, ffn_hidden=4096):
        super().__init__()
        self.xattn = CrossAttention(dim_q, dim_kv, dim_out, num_heads, dropout)
        self.ffn = FFN(dim_out, hidden=ffn_hidden, dropout=dropout)

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        x = self.xattn(Q, K, V, key_padding_mask, attn_mask)
        x = self.ffn(x)
        return x
