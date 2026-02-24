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

"""Processor step for renaming observation keys in an EnvTransition.

NOTE: This class is kept for checkpoint backward compatibility. Saved pipeline JSONs
from earlier checkpoints include a 'rename_observations_processor' step entry.
New pipelines do not include this step since camera keys are always integer-indexed
(observation.images.0, observation.images.1, …) and no runtime renaming is needed.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep


@dataclass
class RenameObservationsProcessorStep(ProcessorStep):
    """Rename keys in the observation dict of an EnvTransition.

    Maps old observation keys to new ones according to ``rename_map``.
    Keys absent from the map are preserved unchanged.

    Attributes:
        rename_map: Mapping from old key names to new key names.
    """

    _registry_name = "rename_observations_processor"

    rename_map: dict[str, str] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        if not self.rename_map:
            return new_transition

        processed_obs = {}
        for key, value in observation.items():
            new_key = self.rename_map.get(key, key)
            processed_obs[new_key] = value
        new_transition[TransitionKey.OBSERVATION] = processed_obs
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Rename top-level keys in a statistics dictionary using the provided mapping.

    Useful for keeping normalization statistics consistent with renamed observation keys.

    Args:
        stats: Nested statistics dict (e.g. {"observation.state": {"mean": ...}}).
        rename_map: Mapping from old feature names to new feature names.

    Returns:
        New statistics dict with top-level keys renamed.
    """
    if not stats:
        return {}
    return {rename_map.get(k, k): deepcopy(v) if v is not None else {} for k, v in stats.items()}
