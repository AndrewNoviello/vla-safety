"""Latent Safety Filter — runtime deployment module.

Implements the safety filtering law from the paper (Equation 9):

    a_exec = π_task(z_t)      if V(ẑ_{t+1}) > ε      (safe → pass through)
             π_shield(z_t)     otherwise               (doomed → override)

At each control step:
  1. Encode the real camera image + proprioception into the world model's
     latent space, maintaining a sliding window of `num_hist` latent frames.
  2. Simulate one step forward in the world model's "imagination" using the
     base policy's proposed action.
  3. Evaluate the safety value function V on the predicted next latent state.
     V tells us whether the robot is *doomed* to eventually fail, even if it
     tries its hardest to recover — not just whether failure is happening now.
  4. If V > ε (safe): let the base policy's action through.
     If V ≤ ε (doomed): override with the safety policy's action.

This module is pure PyTorch — no ROS dependency. The ROS integration lives
in deploy_ros.py which calls this filter.

Usage:
    sf = LatentSafetyFilter(
        wm_checkpoint="outputs/dino_wm_v2/checkpoints/latest/model.pt",
        ddpg_actor_checkpoint="outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt",
        ddpg_critic_checkpoint="outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt",
        stats_path="stats.json",
    )
    sf.reset()

    # In control loop:
    action_raw, info = sf.step(image_rgb_224, proprio_raw, proposed_action_raw)
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T

from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.decoder import Decoder
from dino_wm.visual_world_model import VWorldModel
from latentsafe.ddpg_safety import SafetyActor, SafetyCritic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# World model construction (standalone, no dependency on train.py globals)
# ---------------------------------------------------------------------------

def _build_world_model(
    img_size: int = 224,
    encoder_name: str = "dinov2_vits14",
    num_hist: int = 2,
    num_pred: int = 1,
    action_dim: int = 6,
    proprio_dim: int = 6,
    action_emb_dim: int = 10,
    proprio_emb_dim: int = 10,
    concat_dim: int = 1,
    num_action_repeat: int = 1,
    num_proprio_repeat: int = 1,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.0,
    predictor_emb_dropout: float = 0.0,
    decoder_channel: int = 384,
    decoder_n_res_block: int = 4,
    decoder_n_res_channel: int = 128,
    failure_head_hidden_dim: int = 256,
) -> VWorldModel:
    """Build a VWorldModel with failure_head=True, matching training architecture."""
    encoder = DinoV2Encoder(name=encoder_name)
    emb_dim = encoder.emb_dim  # 384 for dinov2_vits14

    # Freeze encoder (always frozen at deployment)
    for p in encoder.parameters():
        p.requires_grad = False

    # Number of visual patches: img resized to (img_size//16)*patch_size before
    # encoding, giving (img_size//16)^2 patches.
    decoder_scale = 16
    num_vis_patches = (img_size // decoder_scale) ** 2  # 196 for img_size=224

    transition = TransitionModel(
        num_patches=num_vis_patches,
        num_frames=num_hist,
        emb_dim=emb_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        proprio_emb_dim=proprio_emb_dim,
        action_emb_dim=action_emb_dim,
        concat_dim=concat_dim,
        num_proprio_repeat=num_proprio_repeat,
        num_action_repeat=num_action_repeat,
        depth=predictor_depth,
        heads=predictor_heads,
        mlp_dim=predictor_mlp_dim,
        dropout=predictor_dropout,
        emb_dropout=predictor_emb_dropout,
    )

    decoder = Decoder(
        channel=decoder_channel,
        n_res_block=decoder_n_res_block,
        n_res_channel=decoder_n_res_channel,
        emb_dim=emb_dim,
    )

    model = VWorldModel(
        image_size=img_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        transition=transition,
        decoder=decoder,
        proprio_dim=proprio_emb_dim,
        action_dim=action_emb_dim,
        concat_dim=concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
        use_failure_head=True,
        failure_head_hidden_dim=failure_head_hidden_dim,
    )
    return model


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------

def _load_stats(path: str | Path) -> dict:
    """Load dataset statistics from stats.json.

    Returns dict with keys like "observation.state", "action", each containing
    numpy arrays for "mean" and "std".
    """
    with open(path) as f:
        raw = json.load(f)
    stats = {}
    for key, sub in raw.items():
        stats[key] = {}
        for k, v in sub.items():
            if v is not None:
                stats[key][k] = np.array(v, dtype=np.float32)
    return stats


# ---------------------------------------------------------------------------
# LatentSafetyFilter
# ---------------------------------------------------------------------------

class LatentSafetyFilter:
    """Runtime safety filter using latent-space HJ reachability.

    Loads a frozen world model (with failure classifier) and a trained safety
    DDPG (actor + critic). At each step, checks whether the base policy's
    proposed action would lead to a doomed state, and overrides if so.

    Args:
        wm_checkpoint:           Path to VWorldModel weights (includes failure_head).
        ddpg_actor_checkpoint:   Path to SafetyActor weights.
        ddpg_critic_checkpoint:  Path to SafetyCritic weights.
        stats_path:              Path to dataset stats.json (for normalization).
        device:                  'cuda' or 'cpu'.
        epsilon:                 Safety threshold. V > ε means safe. Paper uses 0.3.
        num_hist:                Number of history frames the world model expects.
        img_size:                Image resolution for the world model (224).
        action_dim:              Robot action dimensionality (6 for SO101).
        proprio_dim:             Proprioception dimensionality (6 for SO101).
        **kwargs:                Additional world model config (see _build_world_model).
    """

    def __init__(
        self,
        wm_checkpoint: str,
        ddpg_actor_checkpoint: str,
        ddpg_critic_checkpoint: str,
        stats_path: str,
        device: str = "cuda",
        epsilon: float = 0.3,
        num_hist: int = 2,
        img_size: int = 224,
        action_dim: int = 6,
        proprio_dim: int = 6,
        action_emb_dim: int = 10,
        proprio_emb_dim: int = 10,
        concat_dim: int = 1,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_mlp_dim: int = 2048,
        failure_head_hidden_dim: int = 256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.num_hist = num_hist
        self.img_size = img_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        # Predictor output dimension = encoder_emb_dim + (action+proprio) extras
        # For dinov2_vits14 with concat_dim=1: 384 + (10+10)*1 = 404
        self.predictor_dim = 384 + (action_emb_dim + proprio_emb_dim) * concat_dim

        # ------------------------------------------------------------------
        # 1. Load dataset stats (for normalizing proprio and actions)
        # ------------------------------------------------------------------
        raw_stats = _load_stats(stats_path)
        self.action_mean = torch.tensor(raw_stats["action"]["mean"], device=self.device)
        self.action_std = torch.tensor(raw_stats["action"]["std"], device=self.device)
        self.proprio_mean = torch.tensor(
            raw_stats["observation.state"]["mean"], device=self.device
        )
        self.proprio_std = torch.tensor(
            raw_stats["observation.state"]["std"], device=self.device
        )
        logger.info(f"Loaded stats: action_mean={self.action_mean}, proprio_mean={self.proprio_mean}")

        # ------------------------------------------------------------------
        # 2. Build and load world model (frozen, with failure_head)
        # ------------------------------------------------------------------
        logger.info(f"Building world model from {wm_checkpoint}")
        self.wm = _build_world_model(
            img_size=img_size,
            num_hist=num_hist,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_emb_dim=action_emb_dim,
            proprio_emb_dim=proprio_emb_dim,
            concat_dim=concat_dim,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            predictor_mlp_dim=predictor_mlp_dim,
            failure_head_hidden_dim=failure_head_hidden_dim,
        )
        sd = torch.load(wm_checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = self.wm.load_state_dict(sd, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys in WM checkpoint: {unexpected[:5]}")
        non_fh_missing = [k for k in missing if "failure_head" not in k]
        if non_fh_missing:
            logger.warning(f"Missing WM keys (non-failure-head): {non_fh_missing[:5]}")
        self.wm.to(self.device)
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False
        logger.info("World model loaded and frozen.")

        # ------------------------------------------------------------------
        # 3. Load safety DDPG actor (π_shield) and critic (V)
        # ------------------------------------------------------------------
        logger.info(f"Loading safety actor from {ddpg_actor_checkpoint}")
        self.actor = SafetyActor(
            obs_dim=self.predictor_dim, action_dim=action_dim
        ).to(self.device)
        self.actor.load_state_dict(
            torch.load(ddpg_actor_checkpoint, map_location=self.device, weights_only=True)
        )
        self.actor.eval()

        logger.info(f"Loading safety critic from {ddpg_critic_checkpoint}")
        self.critic = SafetyCritic(
            obs_dim=self.predictor_dim, action_dim=action_dim
        ).to(self.device)
        self.critic.load_state_dict(
            torch.load(ddpg_critic_checkpoint, map_location=self.device, weights_only=True)
        )
        self.critic.eval()

        # ------------------------------------------------------------------
        # 4. Image preprocessing (resize to model input size)
        # ------------------------------------------------------------------
        self._resize = T.Resize((img_size, img_size), antialias=True)

        # ------------------------------------------------------------------
        # 5. Runtime state (set by reset())
        # ------------------------------------------------------------------
        self._z_hist: torch.Tensor | None = None  # (1, ≤num_hist, P, D)
        self._last_action_norm: torch.Tensor | None = None  # (1, 1, action_dim)
        self._override_count = 0
        self._step_count = 0

        logger.info(
            f"LatentSafetyFilter ready: predictor_dim={self.predictor_dim}, "
            f"epsilon={epsilon}, num_hist={num_hist}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        """Raw joint angles → mean-std normalized."""
        return (action_raw - self.action_mean) / (self.action_std + 1e-8)

    def _unnormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """Mean-std normalized → raw joint angles."""
        return action_norm * self.action_std + self.action_mean

    def _normalize_proprio(self, proprio_raw: torch.Tensor) -> torch.Tensor:
        """Raw joint positions → mean-std normalized."""
        return (proprio_raw - self.proprio_mean) / (self.proprio_std + 1e-8)

    def _safety_action_to_raw(self, safety_action: torch.Tensor) -> torch.Tensor:
        """Convert safety DDPG output ([-1, 1]) to raw joint angles.

        The DDPG was trained with actions in [-1, 1]. We interpret these as
        approximately normalized values and unnormalize them. This maps [-1, 1]
        to [mean - std, mean + std], which is a conservative ±1σ range — 
        appropriate for a safety controller making gentle corrections.
        """
        return safety_action * self.action_std + self.action_mean

    # ------------------------------------------------------------------
    # Latent encoding
    # ------------------------------------------------------------------

    def _pool_z(self, z: torch.Tensor) -> torch.Tensor:
        """Mean-pool patch features from the last timestep.

        Args:
            z: (1, T, num_patches, predictor_dim)
        Returns:
            (1, predictor_dim) — flat observation for the DDPG.
        """
        return z[:, -1].mean(dim=1)  # (1, P, D) → mean over patches → (1, D)

    @torch.no_grad()
    def _encode_observation(
        self,
        image_rgb: torch.Tensor,
        proprio_norm: torch.Tensor,
        action_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one real observation into the world model's latent space.

        Args:
            image_rgb:   (1, 1, 3, img_size, img_size) float [0, 1]
            proprio_norm: (1, 1, proprio_dim) mean-std normalized
            action_norm:  (1, 1, action_dim) mean-std normalized

        Returns:
            z: (1, 1, num_patches, predictor_dim)
        """
        obs = {
            "visual": image_rgb.to(self.device),
            "proprio": proprio_norm.to(self.device),
        }
        act = action_norm.to(self.device)
        return self.wm.encode(obs, act)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear latent history. Call when the task episode resets."""
        self._z_hist = None
        self._last_action_norm = torch.zeros(
            1, 1, self.action_dim, device=self.device
        )
        self._override_count = 0
        self._step_count = 0
        logger.info("Safety filter reset.")

    @torch.no_grad()
    def step(
        self,
        image_rgb_224: np.ndarray,
        proprio_raw: np.ndarray,
        proposed_action_raw: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Run one safety filtering step.

        This is the core method called at each control timestep. It encodes the
        real observation, checks safety of the proposed action, and either passes
        it through or overrides with the safety policy.

        Args:
            image_rgb_224:       (H, W, 3) uint8 RGB image at any resolution.
                                 Will be resized to img_size × img_size internally.
            proprio_raw:         (proprio_dim,) raw joint positions (degrees).
            proposed_action_raw: (action_dim,) raw joint commands from the base policy.

        Returns:
            action_raw: (action_dim,) the action to actually execute on the robot.
            info: dict with diagnostic fields:
                - 'safety_value':   V(ẑ_{t+1}), the safety value of the predicted next state
                - 'failure_score':  immediate failure classifier score (negative = failing now)
                - 'overridden':     True if safety policy replaced the base action
                - 'step':           timestep counter
                - 'warmup':         True if history is still filling up (no safety check yet)
        """
        self._step_count += 1

        # --- 1. Preprocess inputs → tensors on device ---
        # Image: uint8 HWC → float CHW [0,1], resize, add batch+time dims
        img_t = torch.from_numpy(image_rgb_224).float().permute(2, 0, 1) / 255.0
        img_t = self._resize(img_t)                        # (3, 224, 224)
        img_t = img_t.unsqueeze(0).unsqueeze(0)            # (1, 1, 3, 224, 224)

        # Proprio: normalize
        proprio_t = torch.tensor(proprio_raw, dtype=torch.float32, device=self.device)
        proprio_norm = self._normalize_proprio(proprio_t)
        proprio_norm = proprio_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, proprio_dim)

        # Proposed action: normalize (for world model encoding)
        proposed_t = torch.tensor(proposed_action_raw, dtype=torch.float32, device=self.device)
        proposed_norm = self._normalize_action(proposed_t)
        proposed_norm_3d = proposed_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)

        # --- 2. Encode real observation into latent space ---
        # Use the *previously executed* action (what led to this observation)
        z_new = self._encode_observation(img_t, proprio_norm, self._last_action_norm)
        # z_new: (1, 1, num_patches, predictor_dim)

        # --- 3. Update latent history (sliding window) ---
        if self._z_hist is None:
            self._z_hist = z_new
        else:
            self._z_hist = torch.cat([self._z_hist, z_new], dim=1)
            # Keep only the last num_hist frames
            if self._z_hist.shape[1] > self.num_hist:
                self._z_hist = self._z_hist[:, -self.num_hist:]

        # --- 4. Warmup: if history isn't full yet, pass through ---
        if self._z_hist.shape[1] < self.num_hist:
            self._last_action_norm = proposed_norm_3d
            return proposed_action_raw, {
                "safety_value": float("inf"),
                "failure_score": 0.0,
                "overridden": False,
                "step": self._step_count,
                "warmup": True,
            }

        # --- 5. One-step safety check in the world model's imagination ---
        #
        # Replace actions in the entire history window with the proposed action
        # (this broadcasts the (1,1,D) action across all T frames — consistent
        # with how the safety DDPG was trained in wm_env.py).
        z_check = self.wm.replace_actions_from_z(
            self._z_hist.clone(), proposed_norm_3d
        )

        # Predict next latent state
        z_pred = self.wm.predict(z_check)       # (1, num_hist, P, D)
        z_next = z_pred[:, -1:]                  # (1, 1, P, D)

        # Pool the predicted next state → flat vector for the DDPG
        obs_next = self._pool_z(z_pred)           # (1, predictor_dim)

        # Evaluate V(ẑ_{t+1}) = Q(ẑ_{t+1}, π_shield(ẑ_{t+1}))
        safety_action_next = self.actor(obs_next)  # (1, action_dim) in [-1, 1]
        safety_value = self.critic(obs_next, safety_action_next).item()

        # Also get the immediate failure classifier score (for logging)
        failure_score = self.wm.predict_failure(z_next).item()

        # --- 6. Filter decision ---
        if safety_value > self.epsilon:
            # SAFE: let the base policy's action through
            executed_action_raw = proposed_action_raw
            overridden = False
        else:
            # DOOMED: override with safety policy from the *current* state
            obs_current = self._pool_z(self._z_hist)          # (1, predictor_dim)
            safety_action_current = self.actor(obs_current)    # (1, action_dim) in [-1, 1]
            executed_action_raw = (
                self._safety_action_to_raw(safety_action_current.squeeze(0))
                .cpu().numpy()
            )
            overridden = True
            self._override_count += 1
            logger.info(
                f"SAFETY OVERRIDE at step {self._step_count}: "
                f"V={safety_value:.3f} < ε={self.epsilon}"
            )

        # --- 7. Remember the executed action for next encoding step ---
        executed_t = torch.tensor(
            executed_action_raw, dtype=torch.float32, device=self.device
        )
        self._last_action_norm = self._normalize_action(executed_t).unsqueeze(0).unsqueeze(0)

        info = {
            "safety_value": safety_value,
            "failure_score": failure_score,
            "overridden": overridden,
            "step": self._step_count,
            "warmup": False,
            "total_overrides": self._override_count,
        }
        return (
            executed_action_raw if isinstance(executed_action_raw, np.ndarray)
            else executed_action_raw,
            info,
        )
