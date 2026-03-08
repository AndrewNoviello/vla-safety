"""Minimal script that loops through a dataloader, mimicking the flow from pi0_train.py."""

import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle

# =====================================================================
# Configuration -- matches pi0_train.py
# =====================================================================

DATASET_REPO_ID = "AndrewNoviello/domino-success-v2"
BATCH_SIZE = 32
NUM_WORKERS = 8
STEPS = 100
PI0_CHUNK_SIZE = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    delta_indices = {"action": list(range(PI0_CHUNK_SIZE))}
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        delta_indices=delta_indices,
        image_transforms=None,
    )

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

    load_times = []
    for step in range(STEPS):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        t1 = time.perf_counter()
        load_time_ms = (t1 - t0) * 1000
        load_times.append(load_time_ms)
        print(f"step {step + 1}/{STEPS}: batch load {load_time_ms:.1f} ms")

    if load_times:
        print(f"\n--- Batch load benchmark ---")
        print(f"  mean: {sum(load_times) / len(load_times):.1f} ms")
        print(f"  min:  {min(load_times):.1f} ms")
        print(f"  max:  {max(load_times):.1f} ms")


if __name__ == "__main__":
    main()
