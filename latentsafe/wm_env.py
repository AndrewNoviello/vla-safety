"""World Model Gym Environment.

Wraps a frozen VWorldModel (with failure_head) as a gymnasium environment.
The RL agent trains entirely inside predicted latent rollouts — no real robot
required.

Architecture mirrors reference Franka_DINOWM_Env
(PyHJ/reach_rl_gym_envs/franka-DINOwm.py, CMU-IntentLab/latent-safety dino).

Observation
-----------
Mean-pooled patch features from the last predicted latent timestep:
    shape (predictor_dim,)
where predictor_dim = encoder_emb_dim + action_emb_dim + proprio_emb_dim
(for concat_dim=1).

Action
------
Raw robot action, clipped to [-1, 1].

Reward
------
tanh(2 * failure_score)  at the last predicted timestep.
  > 0 → predicted safe
  < 0 → predicted unsafe

Rollout mechanics (pure latent space after initial encode)
-----------------------------------------------------------
reset():
    Sample num_hist consecutive frames from dataset.
    Encode raw images + actions → z_hist  (1, num_hist, P, D).

step(action):
    1. Replace last-timestep action in z_hist with new action.
    2. Predict next latent: z_pred = wm.predict(z_hist).
    3. Extract new frame: z_new = z_pred[:, -1:]
    4. Slide history: z_hist = cat(z_hist[:, 1:], z_new)
    5. reward = tanh(2 * failure_score(z_pred[:, -1:]))
    6. obs = z_pred[:, -1].mean(dim=2)   shape (predictor_dim,)
"""

import random
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torchvision.transforms import v2 as T

from dino_wm.visual_world_model import VWorldModel
from data.lerobot_dataset import LeRobotDataset
from data.utils import POLICY_FEATURES, dataset_to_policy_features
from utils.types import FeatureType, NormalizationMode
from utils.processor_utils import normalize


_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE:  NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}

_IMG_TRANSFORM = T.Resize((224, 224), antialias=True)


def _detect_image_key(features: dict) -> str:
    preferred = ("image0", "observation.image", "observation.images.front")
    for k in preferred:
        if k in features and features[k]["dtype"] == "image":
            return k
    for k, v in features.items():
        if v["dtype"] == "image":
            return k
    raise ValueError("No image observation key found in dataset features.")


class WorldModelEnv(gym.Env):
    """Gymnasium environment backed by a frozen VWorldModel.

    Parameters
    ----------
    wm : VWorldModel
        Trained world model with use_failure_head=True. Should already be
        on `device` and in eval mode.
    dataset : LeRobotDataset
        Source of initial-condition windows for reset().
    device : torch.device
    action_dim : int
        Dimension of the robot action space.
    predictor_dim : int
        Output feature dimension of the predictor (= observation_space size).
    num_hist : int
        Number of history frames the world model expects.
    max_episode_steps : int
        Episode length (matches reference: 10).
    action_low, action_high : float
        Action space bounds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        wm: VWorldModel,
        dataset: LeRobotDataset,
        device: torch.device,
        action_dim: int,
        predictor_dim: int,
        num_hist: int = 2,
        max_episode_steps: int = 10,
        action_low: float = -1.0,
        action_high: float = 1.0,
        frameskip: int = 3,
    ):
        super().__init__()
        assert wm.use_failure_head, "WorldModelEnv requires wm.use_failure_head=True"

        self.wm = wm
        self.dataset = dataset
        self.device = device
        self.num_hist = num_hist
        self.max_episode_steps = max_episode_steps
        self.frameskip = frameskip
        self._policy_features = dataset_to_policy_features(POLICY_FEATURES)
        self._image_key = _detect_image_key(POLICY_FEATURES)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(predictor_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Internal state (set in reset())
        self._z_hist: Optional[torch.Tensor] = None  # (1, num_hist, P, D)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_initial_window(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample num_hist consecutive frames from the dataset.

        Returns
        -------
        visual : (1, num_hist, C, H, W)  ImageNet-normalised float32
        proprio : (1, num_hist, proprio_dim)  normalised float32
        action  : (1, num_hist, action_dim)   normalised float32
        """
        # Pick a random valid start index (leave room for num_hist frames)
        max_start = self.dataset.num_frames - self.num_hist * self.frameskip - 1
        start_idx = random.randint(0, max(0, max_start))

        # Build a tiny batch of num_hist frames by directly indexing the dataset
        visuals, proprios, actions = [], [], []
        for i in range(self.num_hist):
            idx = min(start_idx + i * self.frameskip, self.dataset.num_frames - 1)
            item = self.dataset[idx]
            # item[image_key] may be (C, H, W) for a single frame
            img = item[self._image_key]
            if img.dim() == 3:
                img = img.unsqueeze(0)   # (1, C, H, W)
            img = _IMG_TRANSFORM(img)    # (1, C, H, W)
            visuals.append(img)
            proprios.append(item["observation.state"].unsqueeze(0) if item["observation.state"].dim() == 1
                            else item["observation.state"][:1])
            actions.append(item["action"].unsqueeze(0) if item["action"].dim() == 1
                           else item["action"][:1])

        visual  = torch.stack([v.squeeze(0) for v in visuals], dim=0).unsqueeze(0)   # (1, T, C, H, W)
        proprio = torch.stack([p.squeeze(0) for p in proprios], dim=0).unsqueeze(0)  # (1, T, proprio_dim)
        action  = torch.stack([a.squeeze(0) for a in actions],  dim=0).unsqueeze(0)  # (1, T, action_dim)

        # Normalise using dataset stats
        batch = {
            self._image_key: visual,
            "observation.state": proprio,
            "action": action,
        }
        batch = normalize(batch, self.dataset.stats, self._policy_features, _NORM_MAP)

        return (
            batch[self._image_key].float().to(self.device),
            batch["observation.state"].float().to(self.device),
            batch["action"].float().to(self.device),
        )

    def _obs_from_z(self, z: torch.Tensor) -> np.ndarray:
        """Extract mean-pooled patch features from last timestep.

        Args:
            z: (1, T, num_patches, predictor_dim)
        Returns:
            np.ndarray of shape (predictor_dim,)
        """
        # z[:, -1] → (1, P, D) → mean over patches → (1, D)
        pooled = z[:, -1].mean(dim=1)   # (1, D)
        return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.wm.eval()
        self._step_count = 0

        with torch.no_grad():
            visual, proprio, action = self._sample_initial_window()
            obs = {"visual": visual, "proprio": proprio}
            # Encode initial window → latent history
            self._z_hist = self.wm.encode(obs, action)   # (1, num_hist, P, D)

        obs_np = self._obs_from_z(self._z_hist)
        return obs_np, {}

    def step(self, action: np.ndarray):
        assert self._z_hist is not None, "Call reset() before step()"

        self.wm.eval()
        self._step_count += 1

        # Convert action to tensor (1, 1, action_dim)
        ac_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        ac_tensor = ac_tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, action_dim)

        with torch.no_grad():
            # Replace last-timestep action in the latent history then predict
            z_stepped = self.wm.replace_actions_from_z(self._z_hist, ac_tensor)

            z_pred = self.wm.predict(z_stepped)     # (1, num_hist, P, D)

            # Extract new predicted frame and slide window
            z_new = z_pred[:, -1:]                  # (1, 1, P, D)
            self._z_hist = torch.cat(
                [self._z_hist[:, 1:], z_new], dim=1
            )                                       # (1, num_hist, P, D)

            # Failure reward from last predicted frame
            # tanh(2 * score): positive=safe, negative=unsafe
            failure_score = self.wm.predict_failure(z_pred[:, -1:])  # (1, 1, 1)
            reward = float(torch.tanh(2.0 * failure_score).squeeze())

        obs_np = self._obs_from_z(self._z_hist)

        terminated = False
        truncated  = self._step_count >= self.max_episode_steps
        info = {"failure_score": float(failure_score.squeeze())}

        return obs_np, reward, terminated, truncated, info

    def render(self):
        pass   # Not implemented

    def close(self):
        pass
