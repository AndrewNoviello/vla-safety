"""RL training orchestration for latent-space safety DDPG.

Mirrors scripts/run_training_ddpg-dinowm.py from the reference
(CMU-IntentLab/latent-safety, dino branch).

Training schedule
-----------------
1. Warmup phase  (gamma=0.0, warmup_steps steps):
   Collect experience and fit the value function to immediate rewards only,
   without the Bellman recursion. This stabilises value estimates before
   safety constraints are applied.

2. Training phase (gamma=0.95, train_steps × num_train_epochs steps):
   Full HJ reach-avoid Bellman with safety constraint.

Usage
-----
    python -m latentsafe.train_safety_ddpg \
        --wm_checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
        --dataset_repo_id AndrewNoviello/domino-world-v2 \
        --output_dir outputs/safety_ddpg

The script requires a VWorldModel checkpoint trained with use_failure_head=True
(or a checkpoint from the standard world model training — the failure_head will
be randomly initialised if missing).
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES
from latentsafe.wm_env import WorldModelEnv
from latentsafe.ddpg_safety import SafetyDDPG, SafetyDDPGConfig
from utils.utils import init_logging


logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_world_model(
    checkpoint_path: str,
    action_dim: int,
    proprio_dim: int,
    cfg: DinoWMConfig,
    device: torch.device,
) -> VWorldModel:
    """Load VWorldModel from checkpoint with failure_head enabled."""
    encoder = DinoV2Encoder(name=cfg.encoder_name)
    emb_dim = encoder.emb_dim

    decoder_scale  = 16
    num_vis_patches = (cfg.img_size // decoder_scale) ** 2

    transition = TransitionModel(
        num_patches=num_vis_patches,
        num_frames=cfg.num_hist,
        emb_dim=emb_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        proprio_emb_dim=cfg.proprio_emb_dim,
        action_emb_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_proprio_repeat=cfg.num_proprio_repeat,
        num_action_repeat=cfg.num_action_repeat,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        dropout=cfg.predictor_dropout,
        emb_dropout=cfg.predictor_emb_dropout,
    )

    decoder = Decoder(
        channel=cfg.decoder_channel,
        n_res_block=cfg.decoder_n_res_block,
        n_res_channel=cfg.decoder_n_res_channel,
        emb_dim=emb_dim,
    )

    model = VWorldModel(
        image_size=cfg.img_size,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        encoder=encoder,
        transition=transition,
        decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
        use_failure_head=True,
        failure_head_hidden_dim=cfg.failure_head_hidden_dim,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logging.warning(f"Unexpected checkpoint keys: {unexpected}")
    if any("failure_head" not in k for k in missing):
        logging.warning(f"Missing non-failure-head keys: {[k for k in missing if 'failure_head' not in k]}")
    if any("failure_head" in k for k in missing):
        logging.info("failure_head randomly initialised (not in checkpoint)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Allow failure_head to remain trainable if needed (but we freeze here)

    return model.to(device)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def _make_env(
    wm: VWorldModel,
    dataset: LeRobotDataset,
    device: torch.device,
    action_dim: int,
    predictor_dim: int,
    num_hist: int,
    frameskip: int,
    max_episode_steps: int = 10,
) -> WorldModelEnv:
    return WorldModelEnv(
        wm=wm,
        dataset=dataset,
        device=device,
        action_dim=action_dim,
        predictor_dim=predictor_dim,
        num_hist=num_hist,
        max_episode_steps=max_episode_steps,
        frameskip=frameskip,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(output_dir: Path, epoch: int, policy: SafetyDDPG) -> None:
    ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.actor.state_dict(),  ckpt_dir / "actor.pt")
    torch.save(policy.critic.state_dict(), ckpt_dir / "critic.pt")
    logging.info(f"Saved checkpoint to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    wm_checkpoint: str,
    dataset_repo_id: str,
    output_dir: str = "outputs/safety_ddpg",
    classifier_checkpoint: str | None = None,
    # World model config (must match checkpoint)
    num_hist: int = 2,
    num_pred: int = 1,
    frameskip: int = 3,
    img_size: int = 224,
    encoder_name: str = "dinov2_vits14",
    action_emb_dim: int = 10,
    proprio_emb_dim: int = 10,
    concat_dim: int = 1,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.1,
    failure_head_hidden_dim: int = 256,
    # RL schedule (mirrors reference)
    warmup_steps: int = 10_000,
    train_steps_per_epoch: int = 40_000,
    num_train_epochs: int = 15,
    # DDPG hyperparams
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    tau: float = 0.005,
    gamma_train: float = 0.95,
    exploration_noise: float = 0.1,
    buffer_capacity: int = 40_000,
    batch_size: int = 256,
    learning_starts: int = 1_000,
    actor_update_freq: int = 5,
    max_episode_steps: int = 10,
    # Infra
    device_str: str = "cuda",
    log_freq: int = 500,
    seed: int = 42,
) -> None:
    init_logging()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_path / "tb"))

    # --- Dataset ---
    window  = num_hist + num_pred
    indices = [i * frameskip for i in range(window)]
    delta_indices = {}
    for k, v in POLICY_FEATURES.items():
        if v["dtype"] == "image":
            delta_indices[k] = indices
    if "observation.state" in POLICY_FEATURES:
        delta_indices["observation.state"] = indices
    if "action" in POLICY_FEATURES:
        delta_indices["action"] = indices

    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_indices=delta_indices,
        image_transforms=T.Resize((img_size, img_size), antialias=True),
    )

    # --- World model ---
    cfg = DinoWMConfig(
        dataset_repo_id=dataset_repo_id,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        img_size=img_size,
        encoder_name=encoder_name,
        action_emb_dim=action_emb_dim,
        proprio_emb_dim=proprio_emb_dim,
        concat_dim=concat_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        predictor_mlp_dim=predictor_mlp_dim,
        predictor_dropout=predictor_dropout,
        failure_head_hidden_dim=failure_head_hidden_dim,
        use_failure_head=True,
    )

    wm = _load_world_model(wm_checkpoint, action_dim, proprio_dim, cfg, device)

    if classifier_checkpoint is not None:
        clf_sd = torch.load(classifier_checkpoint, map_location="cpu")
        fh_sd = {k: v for k, v in clf_sd.items() if "failure_head" in k}
        wm.load_state_dict(fh_sd, strict=False)
        logging.info(f"Loaded failure_head from classifier checkpoint: {classifier_checkpoint} ({len(fh_sd)} tensors)")

    # predictor_dim = emb_dim + (action_emb_dim + proprio_emb_dim) * concat_dim
    from dino_wm.encoder import DinoV2Encoder as _Enc
    _enc = _Enc(name=encoder_name)
    predictor_dim = _enc.emb_dim + (action_emb_dim + proprio_emb_dim) * concat_dim
    logging.info(f"predictor_dim = {predictor_dim}  action_dim = {action_dim}")

    # --- Gym environment ---
    env = _make_env(
        wm=wm, dataset=dataset, device=device,
        action_dim=action_dim, predictor_dim=predictor_dim,
        num_hist=num_hist, frameskip=frameskip,
        max_episode_steps=max_episode_steps,
    )

    # --- DDPG policy ---
    ddpg_cfg = SafetyDDPGConfig(
        obs_dim=predictor_dim,
        action_dim=action_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=0.0,   # start warmup
        exploration_noise=exploration_noise,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        learning_starts=learning_starts,
        actor_update_freq=actor_update_freq,
    )
    policy = SafetyDDPG(ddpg_cfg, device)

    global_step = 0

    def _collect_steps(n_steps: int, phase: str) -> None:
        nonlocal global_step
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(n_steps):
            action = policy.select_action(obs, add_noise=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            policy.buffer.add(obs, next_obs, action, reward, float(done))

            ep_reward += reward
            obs = next_obs
            global_step += 1

            loss_dict = policy.update()

            if global_step % log_freq == 0 and loss_dict:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"{phase}/{k}", v, global_step)
                writer.add_scalar(f"{phase}/gamma", policy.gamma, global_step)
                logging.info(
                    f"[{phase} step {global_step}] "
                    + "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                    + f"  gamma={policy.gamma}"
                )

            if done:
                writer.add_scalar(f"{phase}/episode_reward", ep_reward, global_step)
                ep_reward = 0.0
                obs, _ = env.reset()

    # ===================================================================
    # Phase 1 — Warmup (gamma=0)
    # ===================================================================
    logging.info(f"=== Warmup phase: {warmup_steps} steps, gamma=0.0 ===")
    policy.gamma = 0.0
    _collect_steps(warmup_steps, phase="warmup")

    # ===================================================================
    # Phase 2 — Training (gamma=gamma_train)
    # ===================================================================
    policy.gamma = gamma_train
    logging.info(f"=== Training phase: {num_train_epochs} epochs × {train_steps_per_epoch} steps, gamma={gamma_train} ===")
    for epoch in range(1, num_train_epochs + 1):
        logging.info(f"--- Epoch {epoch}/{num_train_epochs} ---")
        _collect_steps(train_steps_per_epoch, phase="train")
        _save_checkpoint(output_path, epoch, policy)

    writer.close()
    logging.info("Safety DDPG training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent-space safety DDPG")
    p.add_argument("--wm_checkpoint", required=True,
                   help="Path to VWorldModel model.pt (with or without failure_head)")
    p.add_argument("--classifier_checkpoint", default=None,
                   help="Optional path to classifier_best.pt to load pre-trained failure_head weights")
    p.add_argument("--dataset_repo_id", default="various-and-sundry/domino-world-v3-labeled")
    p.add_argument("--output_dir",      default="outputs/safety_ddpg")
    p.add_argument("--warmup_steps",    type=int,   default=10_000)
    p.add_argument("--train_steps",     type=int,   default=40_000,
                   help="Steps per training epoch")
    p.add_argument("--num_epochs",      type=int,   default=15)
    p.add_argument("--gamma",           type=float, default=0.95)
    p.add_argument("--actor_lr",        type=float, default=1e-4)
    p.add_argument("--critic_lr",       type=float, default=1e-3)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        wm_checkpoint=args.wm_checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        classifier_checkpoint=args.classifier_checkpoint,
        warmup_steps=args.warmup_steps,
        train_steps_per_epoch=args.train_steps,
        num_train_epochs=args.num_epochs,
        gamma_train=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        device_str=args.device,
        seed=args.seed,
    )
