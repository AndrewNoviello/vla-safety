import logging
from pathlib import Path
from typing import Any

import torch
from termcolor import colored
from torch.optim import Optimizer

from utils.types import FeatureType, NormalizationMode
from data.lerobot_dataset import LeRobotDataset
from data.utils import (
    POLICY_FEATURES,
    cycle,
    dataset_to_policy_features,
)
from pi0.config import PI0Config
from pi0.model import PI0Policy
from pi0.processor import preprocess_pi0
from pi0.pretrained import PreTrainedPolicy
from transformers import AutoTokenizer

from utils.optim_utils import cosine_warmup_scheduler
from utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    set_seed,
    update_last_checkpoint,
)
from utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from utils.wandb_utils import WandBLogger

DATASET_REPO_ID = "lerobot/aloha_sim_insertion_scripted"

PRETRAINED_PATH = None

STEPS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 1000
LOG_FREQ = 10
SAVE_CHECKPOINT = True
SAVE_FREQ = 20_000

WANDB_ENABLE = False
WANDB_PROJECT = "lerobot"
WANDB_ENTITY = None
WANDB_NOTES = None

OUTPUT_DIR = 'outputs/aloha_sim_insertion_scripted'

# PI0 chunk size (number of action steps)
PI0_CHUNK_SIZE = 50

def update_policy(
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    device: torch.device,
    lr_scheduler=None,
) -> dict:
    policy.train()

    with torch.amp.autocast(device_type=device.type):
        loss, _ = policy.forward(batch)

    loss.backward()

    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    metrics = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
    }
    return metrics


# Hardcoded normalization for PI0
PI0_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


def train():
    init_logging()

    wandb_logger = None
    if WANDB_ENABLE and WANDB_PROJECT:
        wandb_logger = WandBLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            notes=WANDB_NOTES,
            log_dir=OUTPUT_DIR,
            job_name="pi0",
            policy_type="pi0",
            seed=SEED,
            dataset_repo_id=DATASET_REPO_ID,
        )

    if SEED is not None:
        set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    delta_indices = {"action": list(range(PI0_CHUNK_SIZE))}
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        delta_indices=delta_indices,
        image_transforms=None,
    )

    _features = dataset_to_policy_features(POLICY_FEATURES)
    _input_features = {k: v for k, v in _features.items() if v.type is not FeatureType.ACTION}
    _output_features = {k: v for k, v in _features.items() if v.type is FeatureType.ACTION}
    config = PI0Config(
        input_features=_input_features,
        output_features=_output_features,
        pretrained_path=PRETRAINED_PATH,
        gradient_checkpointing=True,
    )
    if PRETRAINED_PATH:
        policy = PI0Policy.from_pretrained(PRETRAINED_PATH, config=config)
    else:
        policy = PI0Policy(config)
    policy = policy.to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    lr_scheduler = cosine_warmup_scheduler(
        optimizer, STEPS,
        warmup_steps=1_000, decay_steps=30_000, peak_lr=2.5e-5, decay_lr=2.5e-6,
    )
    grad_clip_norm = 1.0

    step = 0
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {OUTPUT_DIR}")
    logging.info(f"steps={STEPS} ({format_big_number(STEPS)}) | dataset: {format_big_number(dataset.num_frames)} frames, {dataset.num_episodes} episodes")
    logging.info(f"Batch size: {BATCH_SIZE} | params: {format_big_number(num_learnable_params)} learnable / {format_big_number(num_total_params)} total")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    # Running averages for logging (sum, count)
    log_sums = {"loss": 0.0, "grad_norm": 0.0}
    log_counts = {k: 0 for k in log_sums}

    all_features = {**policy.input_features, **policy.output_features}
    norm_map = PI0_NORM_MAP

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    for _ in range(step, STEPS):
        batch = next(dl_iter)
        batch = preprocess_pi0(
            batch,
            stats=dataset.stats,
            all_features=all_features,
            norm_map=norm_map,
            tokenizer=tokenizer,
            device=device,
            max_length=48,
            add_batch_dim=False,
        )

        metrics = update_policy(
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            device=device,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        for k in log_sums:
            log_sums[k] += metrics[k]
            log_counts[k] += 1

        is_log_step = LOG_FREQ > 0 and step % LOG_FREQ == 0
        is_saving_step = step % SAVE_FREQ == 0 or step == STEPS

        if is_log_step:
            samples = step * BATCH_SIZE
            avg_loss = log_sums["loss"] / log_counts["loss"]
            avg_grad = log_sums["grad_norm"] / log_counts["grad_norm"]
            logging.info(
                f"step:{format_big_number(step)} smpl:{format_big_number(samples)} "
                f"loss:{avg_loss:.3f} grdn:{avg_grad:.3f}"
            )
            if wandb_logger:
                wandb_logger.log_dict(
                    {"steps": step, "samples": samples, "loss": avg_loss, "grad_norm": avg_grad},
                    step,
                )
            for k in log_sums:
                log_sums[k] = 0.0
                log_counts[k] = 0

        if SAVE_CHECKPOINT and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(OUTPUT_DIR, STEPS, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("End of training")
    policy.push_model_to_hub(DATASET_REPO_ID)


def main():
    train()


if __name__ == "__main__":
    main()
