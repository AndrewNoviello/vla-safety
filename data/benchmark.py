"""Benchmark script to measure dataloader batch loading time."""
import time

import torch
from torchvision import transforms

from data.lerobot_dataset import LeRobotDataset

DATASET_REPO_ID = "various-and-sundry/domino-world-v3-labeled"
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_BATCHES = 20
IMAGE_SIZE = 224


def main():
    image_transforms = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    dataset = LeRobotDataset(DATASET_REPO_ID, image_transforms=image_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    times = []
    for i, batch in enumerate(dataloader):
        if i == 0:
            t0 = time.perf_counter()
            continue
        t1 = time.perf_counter()
        times.append(t1 - t0)
        t0 = t1
        if i >= NUM_BATCHES:
            break

    avg = sum(times) / len(times)
    print(f"Average batch load time: {avg*1000:.1f} ms over {len(times)} batches")
    print(f"Throughput: {BATCH_SIZE / avg:.1f} samples/sec")


if __name__ == "__main__":
    main()
