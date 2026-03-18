"""dino_wm_inference.py — Inference module for the DINO World Model.

Loads weights from HuggingFace Hub (AndrewNoviello/domino-world-wm-v2) and
provides a clean API to:
  - encode observations + actions into latents
  - predict the next latent given a history window
  - decode latents back to images
  - run a full autoregressive rollout

Usage as a library:
    from dino_wm_inference import DinoWMInference
    inf = DinoWMInference()
    z_obses, wm_visual = inf.rollout(obs_0, all_actions)

Standalone sanity check:
    python dino_wm_inference.py [--hf-repo REPO] [--checkpoint FILE] [--device DEVICE]
"""

import sys
from pathlib import Path

import torch

from lerobot.configs.dino_wm_config import DinoWMConfig
from lerobot.datasets.utils import POLICY_FEATURES

# dino_wm_train sets up sys.path for dino_wm *and* exports CFG / helpers.
from dino_wm_train import CFG, DINO_WM_NORM_MAP, _build_model  # noqa: E402

# After dino_wm_train has inserted dino_wm/ into sys.path these are importable.
from models.visual_world_model import VWorldModel  # noqa: E402
from einops import rearrange  # noqa: E402

from huggingface_hub import hf_hub_download, list_repo_files

_ACTION_DIM: int = POLICY_FEATURES["action"]["shape"][-1]         # 6
_PROPRIO_DIM: int = POLICY_FEATURES["observation.state"]["shape"][-1]  # 6


def _resolve_model_path(
    hf_repo_id: str,
    checkpoint_filename: str,
) -> Path:
    """Download model.pt from HuggingFace Hub (or return from cache).

    If *checkpoint_filename* ends with "latest/model.pt" and that exact path
    is not present in the repo (e.g. symlinks aren't uploaded), this function
    falls back to the highest-numbered checkpoint in the repo.
    """
    # Try the requested filename first.
    try:
        path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=checkpoint_filename,
            repo_type="model",
        )
        return Path(path)
    except Exception:
        pass  # fall through to scanning

    # Scan the repo for model.pt files and take the lexicographically last one
    # (step numbers are zero-padded so this gives the latest checkpoint).
    files = sorted(
        f for f in list_repo_files(hf_repo_id, repo_type="model")
        if f.endswith("model.pt") and "optimizer" not in f
    )
    if not files:
        raise FileNotFoundError(
            f"No model.pt found in {hf_repo_id}. "
            "Pass --checkpoint with an explicit filename."
        )
    chosen = files[-1]
    print(f"  '{checkpoint_filename}' not found; using '{chosen}' instead.")
    path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=chosen,
        repo_type="model",
    )
    return Path(path)


class DinoWMInference:
    """Wraps a trained VWorldModel for inference.

    Weights are downloaded from HuggingFace on first use and cached locally
    by huggingface_hub (~/.cache/huggingface/hub/).

    Args:
        hf_repo_id:           HuggingFace model repository ID.
        checkpoint_filename:  Path within the repo to the model weights file.
        local_model_path:     If given, load from this local .pt file instead.
        cfg:                  DinoWMConfig matching the training run.
        device:               'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        hf_repo_id: str = CFG.hf_model_repo_id,
        checkpoint_filename: str = "checkpoints/latest/model.pt",
        local_model_path: str | None = None,
        cfg: DinoWMConfig = CFG,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.cfg = cfg

        # Build model (exact same architecture as training)
        model, _ = _build_model(cfg, _ACTION_DIM, _PROPRIO_DIM)

        # Resolve path to model weights
        if local_model_path is not None:
            resolved = Path(local_model_path)
            print(f"Loading weights from local path: {resolved}")
        else:
            print(f"Downloading weights from HuggingFace: {hf_repo_id} / {checkpoint_filename}")
            resolved = _resolve_model_path(hf_repo_id, checkpoint_filename)
            print(f"  cached at: {resolved}")

        state_dict = torch.load(resolved, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model: VWorldModel = model
        print(f"Model ready on {self.device}.")

    @torch.no_grad()
    def encode(self, obs: dict, act: torch.Tensor) -> torch.Tensor:
        """Encode observations and actions into a combined latent z.

        Args:
            obs: {
                "visual":  (B, T, 3, H, W)  float [0, 1]
                "proprio": (B, T, proprio_dim)  normalized
            }
            act: (B, T, action_dim)  normalized

        Returns:
            z: (B, T, num_patches [+2 if concat_dim=0],
                      emb_dim [+ action/proprio extras if concat_dim=1])
        """
        obs = {k: v.to(self.device) for k, v in obs.items()}
        act = act.to(self.device)
        return self.model.encode(obs, act)

    @torch.no_grad()
    def predict_next(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Predict the next latent given a window of num_hist encoded frames.

        Args:
            z_hist: (B, num_hist, num_patches, dim)

        Returns:
            (B, 1, num_patches, dim) — predicted next-frame latent
        """
        z_hist = z_hist.to(self.device)
        z_pred = self.model.predict(z_hist)   # (B, num_hist, num_patches, dim)
        return z_pred[:, -1:, ...]            # last (newest) predicted frame

    @torch.no_grad()
    def decode_visual(self, z_obs_visual: torch.Tensor) -> torch.Tensor:
        """Decode visual-only latent tokens back to RGB images.

        Args:
            z_obs_visual: (B, T, num_patches, emb_dim)
                The pure visual part of z (after separate_emb has been called,
                or the visual slice of the combined z).

        Returns:
            (B, T, 3, H, W)  float  (values may be slightly outside [0, 1])
        """
        z = z_obs_visual.to(self.device)
        b, t, _np, _ed = z.shape
        decoded = self.model.decoder(z)   # (B*T, 3, H, W)
        return rearrange(decoded, "(b t) c h w -> b t c h w", b=b, t=t)

    @torch.no_grad()
    def rollout(
        self,
        obs_0: dict,
        all_actions: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """Autoregressive rollout from initial observations.

        At every step the world model predicts the next latent; the real
        next-frame observation is *never* fed back in.

        Args:
            obs_0: {
                "visual":  (1, num_hist, 3, H, W)  float [0, 1]
                "proprio": (1, num_hist, proprio_dim)  normalized
            }
            all_actions: (1, num_hist + K, action_dim)  normalized
                The first num_hist actions correspond to the seed frames in
                obs_0; the remaining K actions drive K predicted steps.

        Returns:
            z_obses:   dict with "visual" and "proprio" tensors,
                       each of shape (1, num_hist + K + 1, ...).
            wm_visual: (1, num_hist + K + 1, 3, H, W)
                       Decoded frames (first num_hist are reconstructions of
                       the seed frames; the rest are pure predictions).
        """
        obs_0 = {k: v.to(self.device) for k, v in obs_0.items()}
        all_actions = all_actions.to(self.device)

        z_obses, _z_full = self.model.rollout(obs_0, all_actions)
        wm_visual = self.decode_visual(z_obses["visual"])
        return z_obses, wm_visual


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Sanity-check DinoWMInference with random inputs (no dataset needed)."
    )
    p.add_argument("--hf-repo", default=CFG.hf_model_repo_id,
                   help="HuggingFace model repo ID")
    p.add_argument("--checkpoint", default="checkpoints/latest/model.pt",
                   help="HF filename or local .pt path to model weights")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    local_pt = args.checkpoint if Path(args.checkpoint).exists() else None
    hf_filename = args.checkpoint if local_pt is None else None

    inf = DinoWMInference(
        hf_repo_id=args.hf_repo,
        checkpoint_filename=hf_filename or "checkpoints/latest/model.pt",
        local_model_path=local_pt,
        device=args.device,
    )

    # Random inputs matching training config
    B = 1
    K = 5   # future steps beyond the seed window
    obs_0 = {
        "visual":  torch.rand(B, CFG.num_hist, 3, CFG.img_size, CFG.img_size),
        "proprio": torch.rand(B, CFG.num_hist, _PROPRIO_DIM),
    }
    all_actions = torch.rand(B, CFG.num_hist + K, _ACTION_DIM)

    print(f"\nRunning rollout with random inputs (num_hist={CFG.num_hist}, K={K}) ...")
    z_obses, wm_visual = inf.rollout(obs_0, all_actions)

    print(f"  z_obses['visual'].shape = {z_obses['visual'].shape}")
    print(f"  z_obses['proprio'].shape = {z_obses['proprio'].shape}")
    print(f"  wm_visual.shape          = {wm_visual.shape}")
    print("\nSanity check passed.")
