"""Lightweight dataset over pre-computed pooled predictor latents."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """Dataset backed by a .pt file produced by precompute_latents.py.

    Each item is a dict with:
        "pooled_pred":   (emb_dim,)  float32 — mean-pooled predictor output at last timestep
        "failure_label": scalar      int64   — 0=safe, 1=unsafe, 2=weakly-unsafe
    """

    def __init__(self, path: str | Path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.pooled_pred   = data["pooled_pred"].float()   # (N, D)
        self.failure_label = data["failure_label"].long()  # (N,)
        self.config        = data.get("config", {})

    def __len__(self) -> int:
        return len(self.pooled_pred)

    def __getitem__(self, idx: int) -> dict:
        return {
            "pooled_pred":   self.pooled_pred[idx],
            "failure_label": self.failure_label[idx],
        }
