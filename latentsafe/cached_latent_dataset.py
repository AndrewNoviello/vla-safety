"""Dataset that serves pre-computed DINO-WM latents directly.

Drop-in replacement for `LeRobotDataset` when training the failure_head on a
fixed world-model checkpoint: instead of re-running DINOv2 + transition.build_z
on every sample, load per-frame `z` tensors written by `scripts/cache_latents.py`
and stack them into the same windowed batches.

Each `episode_NNN.pt` contains:
    z:             (T_ep, tokens, D) fp16 — full latent per frame
    z_mean:        (T_ep, D)         fp16 — patch-pooled latent (unused here)
    failure_label: (T_ep,)       int8   — 1 = unsafe, 0 = safe
    frame_index:   (T_ep,)       int64
    index:         (T_ep,)       int64
    episode_index: scalar        int64

Plus a top-level `manifest.json` with dataset-wide stats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CachedLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        num_hist: int,
        num_pred: int,
        frameskip: int,
        include_cls_token: bool | None = None,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory does not exist: {self.cache_dir}")

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found at {manifest_path}. "
                "Did `python scripts/cache_latents.py` finish successfully?"
            )
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("store_full_patches", False):
            raise ValueError(
                f"Cache at {self.cache_dir} was built without --store_full_patches. "
                "Re-run cache_latents.py with --store_full_patches to use this loader."
            )
        cache_has_cls = bool(manifest.get("include_cls_token", False))
        if include_cls_token is not None and cache_has_cls != include_cls_token:
            expected = "CLS-enabled" if include_cls_token else "patch-only"
            actual = "CLS-enabled" if cache_has_cls else "patch-only"
            raise ValueError(
                f"Cache at {self.cache_dir} is {actual}, but this run expects "
                f"{expected} latents. Re-run scripts/cache_latents.py with a "
                "matching world-model checkpoint."
            )

        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.frameskip = int(frameskip)
        self.window = self.num_hist + self.num_pred
        self.cache_format_version = int(manifest.get("cache_format_version", 1))
        self.include_cls_token = cache_has_cls
        self.cls_token_index = manifest.get("cls_token_index")
        self.num_visual_patches = manifest.get("num_visual_patches")

        episode_paths = sorted(self.cache_dir.glob("episode_*.pt"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode_*.pt files in {self.cache_dir}")

        # Eager load: ~84k frames × P × D × 2 B ≈ 12 GB at fp16, fits in RAM.
        # Loading once in the parent means DataLoader workers share the data via
        # copy-on-write fork (Linux) instead of each rereading from disk.
        self._z_per_ep: dict[int, torch.Tensor] = {}
        self._fl_per_ep: dict[int, torch.Tensor] = {}
        self._ep_len: dict[int, int] = {}
        self._index: list[tuple[int, int]] = []

        P_dim = D_dim = None
        for p in episode_paths:
            ep = int(p.stem.rsplit("_", 1)[1])
            data = torch.load(p, weights_only=True, map_location="cpu")
            z = data["z"]                           # (T_ep, P, D) fp16
            n = int(z.shape[0])
            if P_dim is None:
                P_dim, D_dim = int(z.shape[1]), int(z.shape[2])
            self._z_per_ep[ep] = z
            self._fl_per_ep[ep] = data["failure_label"].to(torch.int64)
            self._ep_len[ep] = n
            self._index.extend([(ep, i) for i in range(n)])

        self.num_frames = len(self._index)
        self.num_episodes = len(self._z_per_ep)
        self.predictor_dim = D_dim
        self.num_tokens = P_dim
        self.num_patches = P_dim
        if self.num_visual_patches is None:
            self.num_visual_patches = P_dim - (1 if self.include_cls_token else 0)
        logger.info(
            f"CachedLatentDataset: {self.num_frames} frames across "
            f"{self.num_episodes} episodes, tokens={P_dim} D={D_dim}, "
            f"include_cls_token={self.include_cls_token}, "
            f"window={self.window} (num_hist={self.num_hist}+num_pred={self.num_pred}), "
            f"frameskip={self.frameskip}"
        )

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        ep, local = self._index[idx]
        n = self._ep_len[ep]
        # Mirror LeRobotDataset._get_query_indices: clamp to [0, ep_len-1].
        win = [max(0, min(n - 1, local + i * self.frameskip)) for i in range(self.window)]
        z = self._z_per_ep[ep][win].float()                  # (window, P, D)
        failure_label = self._fl_per_ep[ep][win[-1]]         # scalar int64
        return {"z": z, "failure_label": failure_label}
