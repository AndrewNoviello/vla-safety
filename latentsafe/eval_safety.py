"""Closed-loop safety evaluation of the trained safety DDPG.

Runs both the safety DDPG (trained with HJ Bellman) and a random baseline
inside the WorldModelEnv, and compares:
- Fraction of steps where failure_score < 0 (predicted unsafe)
- Mean episode reward
- Episode success rate (fraction of episodes with all rewards > 0)

Usage
-----
    python -m latentsafe.eval_safety \
        --wm_checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
        --actor_checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt \
        --dataset_repo_id AndrewNoviello/domino-world-v2 \
        --num_episodes 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2 as T

from dino_wm.config import DinoWMConfig
from dino_wm.decoder import Decoder
from dino_wm.encoder import DinoV2Encoder
from dino_wm.transition import TransitionModel
from dino_wm.visual_world_model import VWorldModel
from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES
from latentsafe.wm_env import WorldModelEnv
from latentsafe.ddpg_safety import SafetyActor
from utils.utils import init_logging

logging.basicConfig(level=logging.INFO)


def _load_wm(checkpoint: str, action_dim: int, proprio_dim: int, cfg: DinoWMConfig, device) -> VWorldModel:
    encoder = DinoV2Encoder(name=cfg.encoder_name)
    emb_dim = encoder.emb_dim
    n_patches = (cfg.img_size // 16) ** 2

    transition = TransitionModel(
        num_patches=n_patches, num_frames=cfg.num_hist, emb_dim=emb_dim,
        proprio_dim=proprio_dim, action_dim=action_dim,
        proprio_emb_dim=cfg.proprio_emb_dim, action_emb_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_proprio_repeat=cfg.num_proprio_repeat, num_action_repeat=cfg.num_action_repeat,
        depth=cfg.predictor_depth, heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim, dropout=0.0, emb_dropout=0.0,
    )
    decoder = Decoder(
        channel=cfg.decoder_channel, n_res_block=cfg.decoder_n_res_block,
        n_res_channel=cfg.decoder_n_res_channel, emb_dim=emb_dim,
    )
    model = VWorldModel(
        image_size=cfg.img_size, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
        encoder=encoder, transition=transition, decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=False, train_predictor=False, train_decoder=False,
        use_failure_head=True, failure_head_hidden_dim=cfg.failure_head_hidden_dim,
    )
    sd = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def _rollout(
    env: WorldModelEnv,
    actor,   # callable (obs_np) → action_np, or None for random
    num_episodes: int,
    device: torch.device,
) -> dict:
    """Run actor for num_episodes, return aggregate metrics."""
    ep_rewards = []
    unsafe_fracs = []
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step_unsafe = 0
        step_total  = 0

        done = False
        while not done:
            if actor is None:
                action = env.action_space.sample()
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = actor(obs_t).squeeze(0).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward  += reward
            step_total += 1
            if info.get("failure_score", 1.0) < 0.0:
                step_unsafe += 1

        ep_rewards.append(ep_reward)
        unsafe_fracs.append(step_unsafe / max(step_total, 1))
        if all_safe := (step_unsafe == 0):
            success_count += 1

    return {
        "mean_episode_reward": float(np.mean(ep_rewards)),
        "std_episode_reward":  float(np.std(ep_rewards)),
        "mean_unsafe_frac":    float(np.mean(unsafe_fracs)),
        "success_rate":        success_count / num_episodes,
        "num_episodes":        num_episodes,
    }


def evaluate(
    wm_checkpoint: str,
    actor_checkpoint: str,
    dataset_repo_id: str,
    num_episodes: int = 100,
    device_str: str = "cuda",
    # WM config
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
    failure_head_hidden_dim: int = 256,
    max_episode_steps: int = 10,
) -> dict:
    init_logging()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    cfg = DinoWMConfig(
        dataset_repo_id=dataset_repo_id,
        num_hist=num_hist, num_pred=num_pred, frameskip=frameskip,
        img_size=img_size, encoder_name=encoder_name,
        action_emb_dim=action_emb_dim, proprio_emb_dim=proprio_emb_dim,
        concat_dim=concat_dim,
        predictor_depth=predictor_depth, predictor_heads=predictor_heads,
        predictor_mlp_dim=predictor_mlp_dim,
        use_failure_head=True, failure_head_hidden_dim=failure_head_hidden_dim,
    )

    action_dim  = POLICY_FEATURES["action"]["shape"][-1]
    proprio_dim = POLICY_FEATURES["observation.state"]["shape"][-1]

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

    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_indices=delta_indices,
        image_transforms=T.Resize((img_size, img_size), antialias=True),
    )

    wm = _load_wm(wm_checkpoint, action_dim, proprio_dim, cfg, device)

    enc_tmp       = DinoV2Encoder(name=encoder_name)
    predictor_dim = enc_tmp.emb_dim + (action_emb_dim + proprio_emb_dim) * concat_dim

    env = WorldModelEnv(
        wm=wm, dataset=dataset, device=device,
        action_dim=action_dim, predictor_dim=predictor_dim,
        num_hist=num_hist, max_episode_steps=max_episode_steps, frameskip=frameskip,
    )

    # --- Safety DDPG actor ---
    actor = SafetyActor(obs_dim=predictor_dim, action_dim=action_dim).to(device)
    actor.load_state_dict(torch.load(actor_checkpoint, map_location=device))
    actor.eval()

    logging.info(f"Evaluating Safety DDPG over {num_episodes} episodes ...")
    safety_results = _rollout(env, actor, num_episodes, device)

    logging.info(f"Evaluating Random baseline over {num_episodes} episodes ...")
    random_results = _rollout(env, None, num_episodes, device)

    logging.info("=== Evaluation Results ===")
    for label, res in [("Safety DDPG", safety_results), ("Random baseline", random_results)]:
        logging.info(
            f"  {label}:  reward={res['mean_episode_reward']:.3f}±{res['std_episode_reward']:.3f}  "
            f"unsafe_frac={res['mean_unsafe_frac']:.3f}  "
            f"success_rate={res['success_rate']:.3f}"
        )

    return {"safety_ddpg": safety_results, "random_baseline": random_results}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_checkpoint",    required=True)
    p.add_argument("--actor_checkpoint", required=True)
    p.add_argument("--dataset_repo_id",  default="AndrewNoviello/domino-world-v2")
    p.add_argument("--num_episodes",     type=int, default=100)
    p.add_argument("--device",           default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        wm_checkpoint=args.wm_checkpoint,
        actor_checkpoint=args.actor_checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        num_episodes=args.num_episodes,
        device_str=args.device,
    )
