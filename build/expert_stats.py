"""Router expert-usage counts for Inkling 2 (INKLING_EXPERT_STATS=<path.npy>), for choosing a two-Spark route.

Rank 0 recomputes each MoE layer's top-k (sigmoid(router logits) + bias, as ops.moe.route selects) and adds them
to a fixed [layers, experts] int64 counter on the GPU (index_add_ into a fixed buffer, so CUDA graphs capture it).
Padding tokens are masked. A daemon thread saves the counts to the path every 60 s (atomic rename).
"""
import os
import threading
import time

import numpy as np
import torch

PATH = os.environ.get("INKLING_EXPERT_STATS")
_buf = None
_next_layer = 0


def register(num_experts: int, rank: int) -> int | None:
    """Called by each MoE layer at construction (in layer order); returns its stats row or None."""
    global _buf, _next_layer
    if not PATH or rank != 0:
        return None
    if _buf is None:
        _buf = torch.zeros((64, num_experts), dtype=torch.int64, device="cuda")
        threading.Thread(target=_writer, daemon=True).start()
    layer = _next_layer
    _next_layer += 1
    return layer


def record(layer: int, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int,
           is_padding: torch.Tensor | None) -> None:
    logits = (x @ weight.t()).float()
    ids = (torch.sigmoid(logits) + bias).topk(topk, dim=-1).indices
    ones = torch.ones_like(ids, dtype=torch.int64)
    if is_padding is not None:
        ones = ones * (~is_padding).to(torch.int64)[:, None]
    _buf[layer].index_add_(0, ids.flatten(), ones.flatten())


def _writer() -> None:
    while True:
        time.sleep(60)
        try:
            counts = _buf[:_next_layer].cpu().numpy()
            tmp = PATH + ".tmp.npy"
            np.save(tmp, counts)
            os.replace(tmp, PATH)
        except Exception:  # never take the server down for statistics
            pass
