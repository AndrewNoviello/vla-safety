#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processor step for adding a batch dimension to environment transition data."""

from dataclasses import dataclass

from torch import Tensor

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep


@dataclass
class AddBatchDimensionProcessorStep(ProcessorStep):
    """Add a batch dimension (size 1) to all relevant parts of an EnvTransition.

    Handles:
    - Actions: 1D tensor → unsqueeze(0)
    - State observations (OBS_STATE, OBS_ENV_STATE): 1D tensor → unsqueeze(0)
    - Image observations (OBS_IMAGE, OBS_IMAGES.*): 3D tensor → unsqueeze(0)
    - Task strings: str → [str]
    - Index tensors (index, task_index): 0D tensor → unsqueeze(0)
    """

    _registry_name = "to_batch_processor"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Add batch dim to action
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, Tensor) and action.dim() == 1:
            new_transition[TransitionKey.ACTION] = action.unsqueeze(0)

        # Add batch dim to observation tensors
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            observation = dict(observation)
            for state_key in [OBS_STATE, OBS_ENV_STATE]:
                if state_key in observation:
                    val = observation[state_key]
                    if isinstance(val, Tensor) and val.dim() == 1:
                        observation[state_key] = val.unsqueeze(0)
            if OBS_IMAGE in observation:
                val = observation[OBS_IMAGE]
                if isinstance(val, Tensor) and val.dim() == 3:
                    observation[OBS_IMAGE] = val.unsqueeze(0)
            for key, val in list(observation.items()):
                if key.startswith(f"{OBS_IMAGES}.") and isinstance(val, Tensor) and val.dim() == 3:
                    observation[key] = val.unsqueeze(0)
            new_transition[TransitionKey.OBSERVATION] = observation

        # Add batch dim to complementary data
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp is not None:
            comp = dict(comp)
            if "task" in comp and isinstance(comp["task"], str):
                comp["task"] = [comp["task"]]
            for idx_key in ("index", "task_index"):
                if idx_key in comp:
                    val = comp[idx_key]
                    if isinstance(val, Tensor) and val.dim() == 0:
                        comp[idx_key] = val.unsqueeze(0)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp

        return new_transition
