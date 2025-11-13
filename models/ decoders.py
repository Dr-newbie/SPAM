# -*- coding: utf-8 -*-
# code/models/decoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZINBDecoder(nn.Module):
    """
    입력: (B, D) → (mu, theta, pi) 각 (B, G)
    """
    def __init__(self, in_dim: int, gene_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden // 2, gene_dim)
        self.theta = nn.Linear(hidden // 2, gene_dim)
        self.pi = nn.Linear(hidden // 2, gene_dim)

    def forward(self, z):
        h = self.backbone(z)
        mu = F.softplus(self.mu(h)) + 1e-4
        theta = F.softplus(self.theta(h)) + 1e-4
        pi = torch.sigmoid(self.pi(h))
        return mu, theta, pi
