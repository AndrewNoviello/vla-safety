"""Safety DDPG for latent-space reach-avoid learning.

Implements DDPG with the Hamilton-Jacobi modified Bellman equation:

    V(x) = min(l(x),  γ · V(x'))

where l(x) is the immediate safety reward (positive=safe, negative=unsafe).

Mirrors PyHJ/policy/modelfree/ddpg_avoid_classical_dinowm.py from the
reference (CMU-IntentLab/latent-safety, dino branch).

Key differences from standard DDPG
------------------------------------
- Critic target uses torch.minimum(rewards, gamma * next_q) instead of
  rewards + gamma * next_q.
- 5 actor update steps per 1 critic update step.
- During exploration, extra noise is injected when critic(obs, action) < 0
  (the agent is estimated to be in an unsafe region — explore more aggressively).
- Warmup phase: gamma=0, so the agent simply fits immediate rewards without
  the Bellman recursion (value-function stabilisation).
- Main phase: gamma=0.95 for full constraint-aware learning.

Actor/Critic architecture
--------------------------
Both are 4-layer MLPs with 512 hidden units (matches reference).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latentsafe.base_policy_avoid import BasePolicyAvoid


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden: int = 512, layers: int = 4) -> nn.Sequential:
    """Build a fully-connected MLP with ReLU activations."""
    dims = [in_dim] + [hidden] * layers + [out_dim]
    modules = []
    for i in range(len(dims) - 1):
        modules.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class SafetyActor(nn.Module):
    """Policy network: obs → action ∈ [-1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = _mlp(obs_dim, action_dim, hidden=hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class SafetyCritic(nn.Module):
    """Value network: (obs, action) → scalar V-value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden=hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0

        self.obs     = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),   dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),          dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.FloatTensor(self.obs[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "actions":  torch.FloatTensor(self.actions[idx]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[idx]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Safety DDPG policy
# ---------------------------------------------------------------------------

@dataclass
class SafetyDDPGConfig:
    obs_dim:           int   = 404          # predictor_dim (384+10+10 for default config)
    action_dim:        int   = 6
    hidden_dim:        int   = 512
    actor_lr:          float = 1e-4
    critic_lr:         float = 1e-3
    tau:               float = 0.005        # soft target update coefficient
    gamma:             float = 0.0          # start with 0 (warmup), set to 0.95 for training
    exploration_noise: float = 0.1
    unsafe_extra_noise: float = 0.3        # extra noise when critic(obs,act) < 0
    actor_update_freq: int   = 5            # actor updates per critic update
    buffer_capacity:   int   = 40_000
    batch_size:        int   = 256
    learning_starts:   int   = 1_000


class SafetyDDPG(BasePolicyAvoid):
    """DDPG with HJ reach-avoid Bellman for latent-space safety learning.

    Args:
        cfg: SafetyDDPGConfig
        device: torch.device
    """

    def __init__(self, cfg: SafetyDDPGConfig, device: torch.device):
        super().__init__(gamma=cfg.gamma)
        self.cfg    = cfg
        self.device = device

        # Networks
        self.actor  = SafetyActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.critic = SafetyCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim).to(device)

        # Target networks
        self.target_actor  = SafetyActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.target_critic = SafetyCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim, cfg.action_dim, device)

        self._critic_update_count = 0

    # ------------------------------------------------------------------
    # BasePolicyAvoid interface
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action (no gradient, inference mode)."""
        with torch.no_grad():
            return self.actor(obs)

    def learn(self, batch: dict) -> dict:
        """One gradient step on critic, and possibly actor + target update.

        Returns dict of scalar losses for logging.
        """
        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"].squeeze(-1)   # (N,)
        dones    = batch["dones"].squeeze(-1)      # (N,)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_q       = self.target_critic(next_obs, next_actions).squeeze(-1)  # (N,)
            # Safety Bellman: min(r, γ·V(s'))
            target_q = self.safety_bellman_target(rewards, next_q, dones)

        current_q  = self.critic(obs, actions).squeeze(-1)   # (N,)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self._critic_update_count += 1
        loss_dict = {"critic_loss": critic_loss.item()}

        # --- Actor update (every `actor_update_freq` critic steps) ---
        if self._critic_update_count % self.cfg.actor_update_freq == 0:
            # Maximise critic value → minimise negative critic
            actor_actions = self.actor(obs)
            actor_loss    = -self.critic(obs, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft target update
            self.soft_update(self.actor,  self.target_actor,  self.cfg.tau)
            self.soft_update(self.critic, self.target_critic, self.cfg.tau)

            loss_dict["actor_loss"] = actor_loss.item()

        return loss_dict

    # ------------------------------------------------------------------
    # Action selection with safety-aware exploration
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Choose action from current policy, optionally with exploration noise.

        Extra noise is added when the critic estimates the current state as
        unsafe (critic value < 0), driving the agent to explore away from
        unsafe regions more aggressively.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0)

        if add_noise:
            noise_std = self.cfg.exploration_noise
            # Detect if we're in an estimated-unsafe region
            with torch.no_grad():
                q_val = self.critic(obs_t, action.unsqueeze(0)).item()
            if q_val < 0.0:
                noise_std = self.cfg.unsafe_extra_noise

            noise  = torch.randn_like(action) * noise_std
            action = (action + noise).clamp(-1.0, 1.0)

        return action.cpu().numpy()

    # ------------------------------------------------------------------
    # Training convenience wrapper
    # ------------------------------------------------------------------

    def update(self) -> Optional[dict]:
        """Sample from buffer and call learn(). Returns None if buffer too small."""
        if len(self.buffer) < self.cfg.learning_starts:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        return self.learn(batch)
