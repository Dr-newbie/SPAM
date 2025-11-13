# -*- coding: utf-8 -*-
# code/utils/loss_util.py
import torch
import torch.nn.functional as F

def zinb_nll(
    x: torch.Tensor,          # (B, G) >= 0  (연속값 허용, count면 정규화/스케일은 호출부에서 결정)
    mu: torch.Tensor,         # (B, G) > 0
    theta: torch.Tensor,      # (B, G) > 0  (dispersion)
    pi: torch.Tensor,         # (B, G) in [0,1]
    mask: torch.Tensor | None = None,  # (B, G) bool
    gene_weight: torch.Tensor | None = None,  # (G,) or (B,G)
    reduction: str = "mean",
    eps: float = 1e-8,
):
    """
    Zero-Inflated Negative Binomial Negative Log Likelihood.
    - x가 연속(로그정규화 등)일 때도 근사 사용. count라면 그대로 사용 가능.
    - mu/theta는 softplus+eps 등 호출부에서 안정화해서 전달 권장.
    """
    # 안전 클램프
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)
    pi = torch.clamp(pi, min=eps, max=1 - eps)

    # NB(x) = C(x+θ-1, x) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    nb_log_prob = t1 + t2 + t3              # log NB(x)

    # ZI mixture
    is_zero = (x <= eps)
    log_prob_zero = torch.log(pi + (1.0 - pi) * torch.exp(nb_log_prob) + eps)
    log_prob_nonzero = torch.log(1.0 - pi + eps) + nb_log_prob
    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)

    nll = -log_prob
    if mask is not None:
        nll = torch.where(mask, nll, torch.zeros_like(nll))

    if gene_weight is not None:
        if gene_weight.dim() == 1:
            nll = nll * gene_weight.view(1, -1)
        else:
            nll = nll * gene_weight

    if reduction == "mean":
        denom = (mask.float().sum() if mask is not None else nll.numel())
        denom = max(float(denom), 1.0)
        return nll.sum() / denom
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"unknown reduction: {reduction}")
