"""Smoke-test CLS-token world-model tensor plumbing without loading DINOv2."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel


class DummyEncoder(nn.Module):
    name = "dummy"
    emb_dim = 8
    patch_size = 1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "patch_tokens": torch.randn(b, 4, self.emb_dim, device=x.device),
            "class_token": torch.randn(b, self.emb_dim, device=x.device),
        }


class DummyDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t = z.shape[0], z.shape[1]
        return torch.zeros(b * t, 3, 16, 16, device=z.device)


def main() -> None:
    transition = TransitionModel(
        num_patches=4,
        num_frames=2,
        emb_dim=8,
        proprio_dim=3,
        action_dim=2,
        proprio_emb_dim=5,
        action_emb_dim=7,
        concat_dim=1,
        num_proprio_repeat=1,
        num_action_repeat=1,
        depth=1,
        heads=1,
        mlp_dim=32,
        include_cls_token=True,
    )
    model = VWorldModel(
        image_size=16,
        num_hist=2,
        num_pred=1,
        encoder=DummyEncoder(),
        transition=transition,
        decoder=DummyDecoder(),
        proprio_dim=5,
        action_dim=7,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        use_failure_head=True,
    )

    obs = {
        "visual": torch.zeros(2, 3, 3, 16, 16),
        "proprio": torch.randn(2, 3, 3),
    }
    act = torch.randn(2, 3, 2)

    z = model.encode(obs, act)
    assert z.shape == (2, 3, 5, 20), z.shape
    assert model.patch_tokens_from_z(z).shape == (2, 3, 4, 20)
    assert model.class_token_from_z(z).shape == (2, 3, 13)

    z_pred = model.predict(z[:, :2])
    assert z_pred.shape == (2, 2, 5, 20), z_pred.shape
    assert model.predict_failure(z_pred[:, -1:]).shape == (2, 1, 1)

    decoded = model.decode(z_pred)
    assert decoded["visual"].shape == (2, 2, 3, 16, 16)
    print("CLS world-model smoke test passed.")


if __name__ == "__main__":
    main()
