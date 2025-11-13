import os, sys, time, numpy as np, anndata as ad, scanpy as sc
import torch, torch.nn as nn, torch.nn.functional as F

import torch, torch.nn as nn, torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 충돌 방지
sys.modules.pop("code", None)

from code.models.Foundations import inf_encoder_factory
from code.models.GCN_update import SpatialGCN
from code.models.gene_encoder import GeneEncoder
from code.models.fusion import CrossModalFusion
from code.models.decoders import ZINBDecoder
from loss_util import zinb_nll
from dataset_utils import (
    load_h5ad_with_preproc,
    load_patch_mapping_csv_xenium,
    make_img_gene_dataloader,
    align_map_ids_to_obs,
    reorder_and_prune_by_obs,
)
from code.models.graph_construction import build_knn_graph


def _load_sd(path):
    import torch
    sd = torch.load(path, map_location="cpu")
    # sd가 {'state_dict': ...} 형태일 수도 있음
    if isinstance(sd, dict) and any(k in sd for k in ["state_dict","model","module"]):
        for k in ["state_dict","model","module"]:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd if isinstance(sd, dict) else sd.state_dict()

def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k,v in sd.items() if k.startswith(prefix)}

def load_pretrained_for_finetune(foundation, gene_encoder, gcn,
                                 pt_img_backbone=None, pt_ig=None, pt_is=None):
    # 1) 이미지 백본/어댑터 → foundation에 주입
    if pt_img_backbone:
        sd = _load_sd(pt_img_backbone)

        # 케이스 A: ImageFoundationAdapter로 저장된 경우
        #   - 'img_encoder.foundation.' 또는 'foundation.' 프리픽스
        cand = [
            _strip_prefix(sd, "img_encoder.foundation."),
            _strip_prefix(sd, "foundation."),
            sd,  # 혹시 바로 backbone 키일 수도 있음
        ]
        loaded = False
        for s in cand:
            try:
                foundation.load_state_dict(s, strict=False)
                print(f"[PT] loaded into foundation (strict=False) from {pt_img_backbone}")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] could not map backbone keys from {pt_img_backbone}")

        # 어댑터의 proj(=Linear)까지 쓰고 싶다면 CrossAttnZINB 쪽에 별도 proj를 두고 매핑 필요
        # (현재 파인튜닝은 foundation만 쓰므로 proj는 생략)

    # 2) img↔gene ckpt → gene_encoder에 주입
    if pt_ig:
        sd = _load_sd(pt_ig)
        cand = [
            _strip_prefix(sd, "gene_encoder."),
            sd
        ]
        loaded = False
        for s in cand:
            try:
                gene_encoder.load_state_dict(s, strict=False)
                print(f"[PT] loaded gene_encoder from {pt_ig} (strict=False)")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] gene_encoder keys not found in {pt_ig}")

    # 3) img↔spatial ckpt → gcn에 주입
    if pt_is:
        sd = _load_sd(pt_is)
        cand = [
            _strip_prefix(sd, "gcn."),
            sd
        ]
        loaded = False
        for s in cand:
            try:
                gcn.load_state_dict(s, strict=False)
                print(f"[PT] loaded GCN from {pt_is} (strict=False)")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print(f"[PT][warn] gcn keys not found in {pt_is}")



# ======= 하이레벨 모델 래퍼 =======
class CrossAttnZINBModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, gcn, dim_img, dim_gene, dim_spot,
                 dim_fuse=512, num_heads=8, dropout=0.1, merge="gated-sum",
                 decoder_hidden=1024, gene_dim=1000):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder
        self.gcn = gcn

        self.proj_img  = nn.Linear(dim_img, dim_fuse)
        self.proj_gene = nn.Linear(dim_gene, dim_fuse)
        self.proj_spot = nn.Linear(dim_spot, dim_fuse)

        self.fusion = CrossModalFusion(dim_img=dim_fuse, dim_spot=dim_fuse, dim_gene=dim_fuse,
                                       dim_out=dim_fuse, num_heads=num_heads, dropout=dropout,
                                       merge=merge)
        self.decoder = ZINBDecoder(in_dim=dim_fuse, gene_dim=gene_dim, hidden=decoder_hidden, dropout=dropout)

    def forward(self, batch, edge_index, edge_weight=None):
        """
        batch: dict with
          - image: (B, C, H, W)
          - expr : (B, G)
          - idx  : (B,) long  — 그래프 노드 인덱스(=spot index in current order)
        """
        x_img = batch["image"]
        x_gene = batch["expr"].float()
        node_idx = batch["idx"].long()

        # 1) enc
        zi = self.img_encoder(x_img)                  # (B, Di)
        zg = self.gene_encoder(x_gene)                # (B, Dg)
        zs_all = self.gcn(node_idx, edge_index, edge_weight)  # (B, Ds) — 미니배치 노드 임베딩 반환되도록 GCN 구현되어 있다고 가정

        # 2) same dim
        zi = self.proj_img(zi)
        zg = self.proj_gene(zg)
        zs = self.proj_spot(zs_all)

        # 3) cross-attn fusion
        z_fused, z_is, z_ig = self.fusion(zi, zs, zg)

        # 4) decode (ZINB params)
        mu, theta, pi = self.decoder(z_fused)
        return mu, theta, pi, {"z_fused": z_fused, "z_is": z_is, "z_ig": z_ig}

# ======= 학습 루프(간단) =======
def train_epoch(model, loader, edge_index, edge_weight, optimizer, amp_dtype=torch.float32, device=torch.device("cpu")):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype != torch.float32))
    loss_sum, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type=="cuda" and amp_dtype!=torch.float32)):
            mu, theta, pi, _ = model(batch, edge_index, edge_weight)
            x = batch["expr"].float()
            loss = zinb_nll(x, mu, theta, pi, reduction="mean")
        if scaler.is_enabled():
            scaler.scale(loss).step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        bs = x.size(0)
        loss_sum += loss.item() * bs
        n += bs
    return loss_sum / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, edge_index, edge_weight, amp_dtype=torch.float32, device=torch.device("cpu")):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    preds, gts = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type=="cuda" and amp_dtype!=torch.float32)):
            mu, theta, pi, _ = model(batch, edge_index, edge_weight)
            x = batch["expr"].float()
            yhat = mu
        preds.append(yhat.detach().cpu()); gts.append(x.detach().cpu())
        mse_sum += F.mse_loss(yhat, x, reduction="sum").item()
        mae_sum += F.l1_loss(yhat, x, reduction="sum").item()
        n += x.numel()

    P = torch.cat(preds, 0).numpy()
    Y = torch.cat(gts, 0).numpy()
    Yc, Pc = Y - Y.mean(0, keepdims=True), P - P.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc**2).sum(0) * (Pc**2).sum(0)) + 1e-8
    pcc = np.where(den > 0, num / den, np.nan)
    return {
        "mse": mse_sum / max(n,1),
        "mae": mae_sum / max(n,1),
        "pcc_mean": float(np.nanmean(pcc)),
        "pcc_median": float(np.nanmedian(pcc)),
    }, P, Y

def main():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--pt_img_backbone", default=None, help="contrastive에서 학습된 이미지 백본/어댑터 ckpt")
    p.add_argument("--pt_ig", default=None, help="img↔gene contrastive(ig_model) ckpt 경로")
    p.add_argument("--pt_is", default=None, help="img↔spatial contrastive(is_model) ckpt 경로")

    p.add_argument("--h5ad", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--id_col", default="spot_id")
    p.add_argument("--enc_name", default="uni_v1",
                   choices=["uni_v1","virchow","virchow2","gigapath","hoptimus0","plip","phikon","conch_v1"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--proj_dim_gene", type=int, default=256)
    p.add_argument("--proj_dim_spot", type=int, default=256)
    p.add_argument("--fuse_dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--merge", default="gated-sum", choices=["gated-sum","concat-proj"])
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_dir", default="./ckpts_cross")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # device & amp
    device = torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device)
    if isinstance(device, str): device = torch.device(device)
    amp_dtype = torch.bfloat16 if (device.type=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    if not args.amp: amp_dtype = torch.float32
    print(f"[Device] {device} AMP={args.amp} dtype={amp_dtype}")

    # data
    adata = load_h5ad_with_preproc(args.h5ad, use_hvg=False, n_top_genes=None,
                                   x_key="x_centroid", y_key="y_centroid")
    map_df = load_patch_mapping_csv_xenium(args.csv, root_dir=args.root)
    map_df = align_map_ids_to_obs(adata, map_df, id_col=args.id_col)
    map_df_aligned, coords_np = reorder_and_prune_by_obs(adata, map_df, id_col=args.id_col, coord_key="spatial")

    loader, gene_dim = make_img_gene_dataloader(
        adata=adata, map_df=map_df_aligned,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_transform=None  # foundation 내부 transform 사용 시 adapter에서 호출하는 구조면 None으로 둠
    )
    edge_index, edge_weight = build_knn_graph(coords_np, k=args.k, use_weight=True)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    # encoders
    Foundation = inf_encoder_factory(args.enc_name)
    foundation = Foundation(weights_path=None)     # img_encoder
    img_dim = foundation.get_output_dim()

    gene_encoder = GeneEncoder(input_dim=gene_dim, model_size_omic="small",
                               proj_dim=args.proj_dim_gene, normalize=False)
    spot_gcn = SpatialGCN(in_dim=args.proj_dim_spot, hidden_dim=256, out_dim=args.proj_dim_spot, dropout=0.2)

    load_pretrained_for_finetune(
        foundation=foundation,
        gene_encoder=gene_encoder,
        gcn=spot_gcn,
        pt_img_backbone=args.pt_img_backbone,
        pt_ig=args.pt_ig,
        pt_is=args.pt_is,
    )

    # 래퍼 모델
    model = CrossAttnZINBModel(
        img_encoder=foundation,
        gene_encoder=gene_encoder,
        gcn=spot_gcn,
        dim_img=img_dim, dim_gene=args.proj_dim_gene, dim_spot=args.proj_dim_spot,
        dim_fuse=args.fuse_dim, num_heads=args.heads, dropout=args.dropout,
        merge=args.merge, decoder_hidden=1024, gene_dim=gene_dim
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"pcc": -1.0, "path": None}
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr = train_epoch(model, loader, edge_index, edge_weight, optim, amp_dtype, device)
        ev, P, Y = eval_epoch(model, loader, edge_index, edge_weight, amp_dtype, device)
        dt = time.time() - t0
        print(f"[{ep:03d}] trainZINB={tr:.6f} | eval MSE={ev['mse']:.6e} MAE={ev['mae']:.6e} "
              f"PCC(mean)={ev['pcc_mean']:.4f} med={ev['pcc_median']:.4f}  ({dt:.1f}s)")
        if ev["pcc_mean"] > best["pcc"]:
            best["pcc"] = ev["pcc_mean"]
            out_dir = os.path.join(args.save_dir, "best")
            os.makedirs(out_dir, exist_ok=True)
            rec = ad.AnnData(X=P, obs=adata.obs.copy(), var=adata.var.copy(), uns=adata.uns.copy())
            rec.write(os.path.join(out_dir, "recovered_data.h5ad"), compression="gzip")
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            best["path"] = out_dir
            print(f"[BEST↑] saved to {out_dir}")

    print(f"Done. best PCC(mean)={best['pcc']:.4f}, path={best['path']}")

if __name__ == "__main__":
    main()
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

"""
python main.py finetune \
  --h5ad /path/section.h5ad \
  --csv  /path/patch_map.csv \
  --root /path \
  --enc_name uni_v1 \
  --epochs 20 --batch_size 128 \
  --k 12 --lr 3e-4 --weight_decay 0.05 \
  --device auto --amp --save_dir ./ckpts_cross \
  --pt_img_backbone ./ckpts/joint/best_img_encoder.pt \
  --pt_ig ./ckpts/joint/best_ig_model.pt \
  --pt_is ./ckpts/joint/best_is_model.pt
"""