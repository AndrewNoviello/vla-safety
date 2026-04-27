"""Transition model: predicts next latent state from (latent, proprio, action).

Bundles proprio and action encoders with the transformer predictor.
Takes latent, proprio, and actions explicitly.
"""

import torch
from torch import nn
from einops import rearrange, repeat


class _VectorEmbedding(nn.Module):
    """Embed 1D vectors (B, T, D) -> (B, T, emb_dim) via Conv1d."""

    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, emb_dim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x.permute(0, 2, 1)


def _generate_mask_matrix(npatch: int, nwindow: int, device: torch.device) -> torch.Tensor:
    zeros = torch.zeros(npatch, npatch, device=device)
    ones = torch.ones(npatch, npatch, device=device)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim, npatch, nwindow, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.npatch = npatch
        self.nwindow = nwindow

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        bias = _generate_mask_matrix(self.npatch, self.nwindow, x.device)
        dots = dots.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(self, dim, npatch, nwindow, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    _Attention(dim, npatch, nwindow, heads=heads, dim_head=dim_head, dropout=dropout),
                    _FeedForward(dim, mlp_dim, dropout=dropout),
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TransitionModel(nn.Module):
    """Transition model: latent, proprio, action -> predicted latent.

    Encodes proprio and action internally, combines with visual latent,
    and runs a causal transformer to predict the next state.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        emb_dim: int,
        proprio_dim: int,
        action_dim: int,
        proprio_emb_dim: int,
        action_emb_dim: int,
        concat_dim: int,
        num_proprio_repeat: int,
        num_action_repeat: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        assert concat_dim in {0, 1}, "concat_dim must be 0 or 1"

        self.concat_dim = concat_dim
        self.num_proprio_repeat = num_proprio_repeat
        self.num_action_repeat = num_action_repeat
        self.proprio_dim = proprio_emb_dim * num_proprio_repeat
        self.action_dim = action_emb_dim * num_action_repeat

        self.proprio_encoder = _VectorEmbedding(in_dim=proprio_dim, emb_dim=proprio_emb_dim)
        self.action_encoder = _VectorEmbedding(in_dim=action_dim, emb_dim=action_emb_dim)

        # Predictor dimension: visual + (proprio+action) if concat_dim=1
        predictor_dim = emb_dim + (
            proprio_emb_dim * num_proprio_repeat + action_emb_dim * num_action_repeat
        ) * concat_dim

        # Number of tokens per frame (concat_dim=0 adds proprio and action as extra tokens)
        num_tokens = num_patches + (2 if concat_dim == 0 else 0)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_tokens, predictor_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            dim=predictor_dim,
            npatch=num_tokens,
            nwindow=num_frames,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def _combine_inputs(
        self,
        latent: torch.Tensor,
        proprio_emb: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Combine visual latent with proprio and action embeddings."""
        if self.concat_dim == 0:
            z = torch.cat(
                [
                    latent,
                    proprio_emb.unsqueeze(2),
                    action_emb.unsqueeze(2),
                ],
                dim=2,
            )
        else:
            f = latent.shape[2]
            proprio_tiled = repeat(
                proprio_emb.unsqueeze(2), "b t 1 a -> b t f a", f=f
            )
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(action_emb.unsqueeze(2), "b t 1 a -> b t f a", f=f)
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat([latent, proprio_repeated, act_repeated], dim=3)
        return z

    def forward(
        self,
        latent: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, T, num_patches, emb_dim) visual patch embeddings
            proprio: (B, T, proprio_dim) raw proprioceptive state
            action: (B, T, action_dim) raw actions

        Returns:
            z_pred: (B, T, num_tokens, predictor_dim) predicted latent
        """
        proprio_emb = self.proprio_encoder(proprio)
        action_emb = self.action_encoder(action)

        z = self._combine_inputs(latent, proprio_emb, action_emb)
        return self.forward_z(z)

    def forward_z(self, z: torch.Tensor) -> torch.Tensor:
        """Run transformer on pre-combined z (e.g. from previous rollout step)."""
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = z + self.pos_embedding[:, : z.shape[1]]
        z = self.dropout(z)
        z = self.transformer(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def build_z(
        self,
        latent: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Combine latent, proprio, action into z (no transformer). For targets/encoding."""
        proprio_emb = self.proprio_encoder(proprio)
        action_emb = self.action_encoder(action)
        return self._combine_inputs(latent, proprio_emb, action_emb)

    def replace_actions_in_z(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Replace the action portion of z with new action embeddings (for rollout)."""
        action_emb = self.action_encoder(action)
        if self.concat_dim == 0:
            z = z.clone()
            z[:, :, -1, :] = action_emb
        else:
            z = z.clone()
            act_tiled = repeat(
                action_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2]
            )
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim :] = act_repeated
        return z
