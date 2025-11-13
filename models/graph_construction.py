import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import coalesce, add_self_loops

def _coalesce(edge_index: torch.Tensor,
              edge_attr: torch.Tensor | None,
              reduce: str = "min"):
    """
    torch_geometric 버전에 따라 시그니처가 다름:
      - 신버전: coalesce(edge_index, edge_attr=None, reduce='add'|'min'...)
      - 구버전: coalesce(edge_index, edge_attr=None, m=None, n=None, reduce='add'...)
    둘 다 지원하도록 try/except로 처리.
    """
    try:
        # 신버전 (권장)
        return coalesce(edge_index, edge_attr, reduce=reduce)
    except TypeError:
        # 구버전
        N = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else 0
        return coalesce(edge_index, edge_attr, N, N, reduce=reduce)

@torch.no_grad()
def build_knn_graph(coords: np.ndarray,
                    k: int = 12,
                    use_weight: bool = True):
    """
    coords: (N,2) array of (x,y). 단위는 임의. 값이 매우 크면 사전에 정규화 권장.
    k: 각 노드에서 self 제외 k이웃. (양방향 간선으로 만듦)
    use_weight: 거리 기반 가우시안 weight 사용.

    Returns:
      edge_index: LongTensor (2, E)
      edge_weight: FloatTensor (E,) or None
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N,2)"
    N = int(coords.shape[0])
    if N == 0:
        return (torch.empty(2, 0, dtype=torch.long),
                None if not use_weight else torch.empty(0, dtype=torch.float32))

    coords = np.asarray(coords, dtype=np.float64)

    # KNN (self 포함해서 k+1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    dists, indices = nbrs.kneighbors(coords)              # (N, k+1)

    # i -> j (self 제외)
    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].reshape(-1)
    dist_vals = dists[:, 1:].reshape(-1)

    # 양방향으로 확장
    row_all = np.concatenate([rows, cols], axis=0)
    col_all = np.concatenate([cols, rows], axis=0)
    dist_all = np.concatenate([dist_vals, dist_vals], axis=0)

    edge_index = torch.tensor(np.stack([row_all, col_all], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(dist_all, dtype=torch.float32)

    # 중복 간선 병합 (거리의 최소 유지)
    edge_index, edge_attr = _coalesce(edge_index, edge_attr, reduce="min")

    # 가우시안 가중치
    if use_weight:
        # sigma: 각 노드 k번째 이웃거리의 중앙값
        sigma = float(np.median(dists[:, -1]))
        if sigma <= 0:
            sigma = 1e-6
        edge_weight = torch.exp(-(edge_attr ** 2) / (2.0 * sigma ** 2))
        # self-loop: 가중치 1.0
        edge_index, edge_weight = add_self_loops(edge_index,
                                                 edge_attr=edge_weight,
                                                 num_nodes=N,
                                                 fill_value=1.0)
    else:
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=N)

    return edge_index, edge_weight
