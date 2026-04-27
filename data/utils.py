from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets
import torch
from datasets import Dataset
from PIL import Image as PILImage
from torchvision import transforms

from utils.types import FeatureType, PolicyFeature
from utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from utils.utils import cast_stats_to_numpy, load_json, suppress_progress_bars

STATS_PATH = "meta/stats.json"

PARQUET_FEATURES = datasets.Features({
    "frame_index": datasets.Value("int64"),
    "observation.state": datasets.Sequence(length=6, feature=datasets.Value("float64")),
    "action": datasets.Sequence(length=6, feature=datasets.Value("float64")),
    "timestamp": datasets.Value("float64"),
    "episode_index": datasets.Value("int64"),
    "index": datasets.Value("int64"),
    "task_index": datasets.Value("int64"),
})

POLICY_FEATURES = {
    "image": {"dtype": "image", "shape": (224, 224, 3)},
    "frame_index": {"dtype": "int64", "shape": (1,)},
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (6,)},
    "timestamp": {"dtype": "float32", "shape": (1,)},
    "episode_index": {"dtype": "int64", "shape": (1,)},
    "index": {"dtype": "int64", "shape": (1,)},
    "task_index": {"dtype": "int64", "shape": (1,)},
}

def load_episode_parquets(
    pq_dir: Path, features: datasets.Features | None = None
) -> Dataset:
    """Load data/episode_*.parquet files and return an HF Dataset."""
    paths = sorted(
        pq_dir.glob("episode_*.parquet"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]) if "_" in p.stem else 0,
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No episode_*.parquet files in {pq_dir}")
    with suppress_progress_bars():
        feats = features if features is not None else PARQUET_FEATURES
        return Dataset.from_parquet([str(p) for p in paths], features=feats)


def load_stats(local_dir: Path) -> dict | None:
    if not (local_dir / STATS_PATH).exists():
        return None
    return cast_stats_to_numpy(load_json(local_dir / STATS_PATH))

def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:
    """Convert a batch from a Hugging Face dataset to torch tensors."""
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is not None:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy features."""
    result: dict[str, PolicyFeature] = {}
    for key, feat in features.items():
        shape = feat["shape"]
        if feat["dtype"] in ["image", "video"]:
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = feat.get("names") or []
            if len(names) >= 3 and names[2] in ["channel", "channels"]:
                shape = (shape[2], shape[0], shape[1])
            result[key] = PolicyFeature(FeatureType.VISUAL, shape)
        elif key == "state":
            result[key] = PolicyFeature(FeatureType.STATE, shape)
        elif key == OBS_ENV_STATE:
            result[key] = PolicyFeature(FeatureType.ENV, shape)
        elif key.startswith("observation.language"):
            result[key] = PolicyFeature(FeatureType.LANGUAGE, shape)
        elif key == ACTION or key.startswith(ACTION):
            result[key] = PolicyFeature(FeatureType.ACTION, shape)
        elif key.startswith(OBS_STR):
            result[key] = PolicyFeature(FeatureType.STATE, shape)
    return result

def cycle(iterable: Any) -> Iterator[Any]:
    """Dataloader-safe cyclical iterator."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
