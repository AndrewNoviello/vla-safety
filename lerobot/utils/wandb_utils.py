#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Any

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def make_wandb_tags(
    policy_type: str,
    seed: int | None = None,
    dataset_repo_id: str | None = None,
    *,
    truncate: bool = False,
    max_tag_length: int = 64,
) -> list[str]:
    """Build a list of W&B tags from training parameters."""

    def _maybe_truncate(tag: str) -> str:
        if not truncate or len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    tags = [_maybe_truncate(f"policy:{policy_type}")]
    if seed is not None:
        tags.append(_maybe_truncate(f"seed:{seed}"))
    if dataset_repo_id is not None:
        tags.append(_maybe_truncate(f"dataset:{dataset_repo_id}"))
    return tags


def make_wandb_group(
    policy_type: str,
    seed: int | None = None,
    dataset_repo_id: str | None = None,
) -> str:
    return "-".join(make_wandb_tags(policy_type, seed, dataset_repo_id))


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    return match.groups(0)[0]


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A helper class to log objects using wandb."""

    def __init__(
        self,
        *,
        project: str = "lerobot",
        entity: str | None = None,
        notes: str | None = None,
        run_id: str | None = None,
        mode: str | None = None,
        disable_artifact: bool = False,
        log_dir: Path,
        job_name: str,
        resume: bool = False,
        policy_type: str,
        seed: int | None = None,
        dataset_repo_id: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ):
        self.disable_artifact = disable_artifact
        self.log_dir = log_dir
        self.job_name = job_name
        self._group = make_wandb_group(policy_type, seed, dataset_repo_id)

        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            run_id
            if run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=project,
            entity=entity,
            name=self.job_name,
            notes=notes,
            tags=make_wandb_tags(policy_type, seed, dataset_repo_id, truncate=True),
            dir=self.log_dir,
            config=config_dict,
            save_code=False,
            job_type="train_eval",
            resume="must" if resume else None,
            mode=mode if mode in ["online", "offline", "disabled"] else "online",
        )
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        for k, v in d.items():
            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d[custom_step_key]
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._wandb.log(data)
                continue

            self._wandb.log(data={f"{mode}/{k}": v}, step=step)
