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
"""
Training script with hardcoded configuration.

Edit the values in the "Configuration" section below, then run:

    python -m lerobot.lerobot_train

Or with accelerate for multi-GPU:

    accelerate launch -m lerobot.lerobot_train
"""
import datetime as dt
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import load_dataset
from lerobot.datasets.augmentation import image_transforms
from lerobot.datasets.utils import cycle, resolve_delta_timestamps
from lerobot.optim import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.processor_pi0 import _ensure_newline
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.tokenizer_processor import TextTokenizer
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.processing import normalize, to_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

# =====================================================================
# Configuration -- edit these values for your experiment
# =====================================================================

DATASET_REPO_ID = "lerobot/aloha_sim_insertion_scripted"
DATASET_ROOT = None
DATASET_EPISODES = None

# Hardcoded PI0 config
POLICY_TYPE = "pi0"
PRETRAINED_PATH = None
PUSH_TO_HUB = False

STEPS = 10
BATCH_SIZE = 1
NUM_WORKERS = 0
SEED = 1000
LOG_FREQ = 10
SAVE_CHECKPOINT = True
SAVE_FREQ = 20_000
TOLERANCE_S = 1e-4

IMAGE_TRANSFORMS_ENABLED = False

PEFT_KWARGS = None  # set to e.g. {"method_type": "LORA", "r": 16} to enable PEFT

WANDB_ENABLE = False
WANDB_PROJECT = "lerobot"
WANDB_ENTITY = None
WANDB_NOTES = None

OUTPUT_DIR = None  # auto-generated if None

# =====================================================================
# Training logic
# =====================================================================


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# Hardcoded normalization for PI0
PI0_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


def train():
    output_dir = OUTPUT_DIR
    if output_dir is None:
        now = dt.datetime.now()
        output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{POLICY_TYPE}"
    else:
        output_dir = Path(output_dir)

    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = False  # Use accelerator default
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if WANDB_ENABLE and WANDB_PROJECT and is_main_process:
        wandb_logger = WandBLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            notes=WANDB_NOTES,
            log_dir=output_dir,
            job_name=POLICY_TYPE,
            policy_type=POLICY_TYPE,
            seed=SEED,
            dataset_repo_id=DATASET_REPO_ID,
        )
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if SEED is not None:
        set_seed(SEED, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- Dataset ---
    image_transforms_fn = image_transforms() if IMAGE_TRANSFORMS_ENABLED else None

    def _create_dataset():
        ds = load_dataset(
            DATASET_REPO_ID,
            episodes=DATASET_EPISODES,
            image_transforms=image_transforms_fn,
            root=DATASET_ROOT,
            tolerance_s=TOLERANCE_S,
        )
        delta_timestamps = resolve_delta_timestamps(POLICY_TYPE, ds)
        if delta_timestamps is not None:
            ds = load_dataset(
                DATASET_REPO_ID,
                episodes=DATASET_EPISODES,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms_fn,
                root=DATASET_ROOT,
                tolerance_s=TOLERANCE_S,
            )
        return ds

    if is_main_process:
        logging.info("Creating dataset")
        dataset = _create_dataset()

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = _create_dataset()

    # --- Policy ---
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        policy_type=POLICY_TYPE,
        ds_meta=dataset.meta,
        pretrained_path=PRETRAINED_PATH,
        use_peft=PEFT_KWARGS is not None,
    )

    is_peft = PEFT_KWARGS is not None
    if is_peft:
        logging.info("Using PEFT! Wrapping model.")
        policy = policy.wrap_with_peft(peft_cli_overrides=PEFT_KWARGS)

    accelerator.wait_for_everyone()

    # --- Optimizer ---
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler, grad_clip_norm = make_optimizer_and_scheduler(
        POLICY_TYPE, policy.parameters(), STEPS
    )

    step = 0
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
        logging.info(f"steps={STEPS} ({format_big_number(STEPS)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = BATCH_SIZE * num_processes
        logging.info(f"Effective batch size: {BATCH_SIZE} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # --- Dataloader ---
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = BATCH_SIZE * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    all_features = {**policy.input_features, **policy.output_features}
    norm_map = PI0_NORM_MAP

    tokenizer = TextTokenizer(
        tokenizer_name="google/paligemma-3b-pt-224",
        max_length=48,
        padding_side="right",
        padding="max_length",
    )

    for _ in range(step, STEPS):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = normalize(batch, dataset.stats, all_features, norm_map)
        batch = _ensure_newline(batch)
        batch = tokenizer(batch)
        batch = to_device(batch, device)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        is_log_step = LOG_FREQ > 0 and step % LOG_FREQ == 0 and is_main_process
        is_saving_step = step % SAVE_FREQ == 0 or step == STEPS

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if SAVE_CHECKPOINT and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(output_dir, STEPS, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    is_peft=is_peft,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")

        if PUSH_TO_HUB:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if is_peft:
                unwrapped_policy.push_model_to_hub(DATASET_REPO_ID, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(DATASET_REPO_ID)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
