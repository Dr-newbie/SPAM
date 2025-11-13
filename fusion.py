import torch
import torch.nn as nn
import torch.nn.functional as F
from code.utils.attention import CrossAttentionBlock

class CrossModalFusion(nn.Module):
    """
    입력:
      z_img  : (B, Di)  — 이미지 임베딩(패치/풀링 후 1토큰이라고 가정)
      z_spot : (B, Ds)  — 공간(GCN) 임베딩(1토큰)
      z_gene : (B, Dg)  — 유전자 임베딩(1토큰; SNN/MLP 인코더 출력)
    처리:
      img<-spot cross-attn, img<-gene cross-attn → 두 토큰을 게이트합/concat+proj
    출력:
      z_fused: (B, D)
    """
    def __init__(self, dim_img: int, dim_spot: int, dim_gene: int,
                 dim_out: int, num_heads: int = 8, dropout: float = 0.1, ffn_hidden: int = 4096,
                 merge: str = "gated-sum"):  # "gated-sum" | "concat-proj"
        super().__init__()
        self.merge = merge
        self.xa_is = CrossAttentionBlock(dim_q=dim_img, dim_kv=dim_spot, dim_out=dim_out,
                                         num_heads=num_heads, dropout=dropout, ffn_hidden=ffn_hidden)
        self.xa_ig = CrossAttentionBlock(dim_q=dim_img, dim_kv=dim_gene, dim_out=dim_out,
                                         num_heads=num_heads, dropout=dropout, ffn_hidden=ffn_hidden)
        if merge == "gated-sum":
            self.gate = nn.Sequential(
                nn.Linear(dim_out * 2, dim_out),
                nn.Sigmoid()
            )
        elif merge == "concat-proj":
            self.proj = nn.Linear(dim_out * 2, dim_out)
        else:
            raise ValueError(f"Unknown merge: {merge}")

    def forward(self, z_img, z_spot, z_gene):
        # (B, D) → (B,1,D)
        Qi = z_img.unsqueeze(1)
        Ks = z_spot.unsqueeze(1)
        Kg = z_gene.unsqueeze(1)

        # img<-spot / img<-gene
        his = self.xa_is(Qi, Ks, Ks)  # (B,1,Dout)
        hig = self.xa_ig(Qi, Kg, Kg)  # (B,1,Dout)

        # merge
        if self.merge == "gated-sum":
            cat = torch.cat([his, hig], dim=-1)     # (B,1,2D)
            gate = self.gate(cat)                   # (B,1,D)
            z = gate * his + (1.0 - gate) * hig     # (B,1,D)
        else:
            z = torch.cat([his, hig], dim=-1)       # (B,1,2D)
            z = self.proj(z)                        # (B,1,D)

        return z.squeeze(1), his.squeeze(1), hig.squeeze(1) 