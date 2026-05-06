from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import InMemoryTable
from PIL import Image as PILImage
from torchvision import transforms

from utils.types import FeatureType, PolicyFeature
from utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from utils.utils import cast_stats_to_numpy, load_json

STATS_PATH = "meta/stats.json"

FLAT_STATE_COLS = [f"observation.state_j{i}" for i in range(1, 7)]
FLAT_ACTION_COLS = [f"action_j{i}" for i in range(1, 7)]
_FLAT_DROP_COLS = (
    FLAT_STATE_COLS
    + FLAT_ACTION_COLS
    + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw", "label"]
)

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


def _episode_index_from_path(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _pack_columns(table: pa.Table, cols: list[str]) -> pa.Array:
    """Stack scalar columns into a FixedSizeList[len(cols), float64] array."""
    arrays = [table.column(c).cast(pa.float64()).to_numpy(zero_copy_only=False) for c in cols]
    stacked = np.stack(arrays, axis=1).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(stacked, type=pa.float64()), len(cols))


def load_episode_parquets(pq_dir: Path) -> Dataset:
    """Load flat-format episode_*.parquet files into an HF Dataset.

    Each input row has scalar columns observation.state_j{1..6}, action_j{1..6},
    ee_*, label, frame_index, timestamp. We pack the joint columns into length-6
    sequence columns and synthesize episode_index/index/task_index/failure_label.
    """
    paths = sorted(pq_dir.glob("episode_*.parquet"), key=_episode_index_from_path)
    if not paths:
        raise FileNotFoundError(f"No episode_*.parquet files in {pq_dir}")

    tables: list[pa.Table] = []
    frame_offset = 0
    for ep_idx, p in enumerate(paths):
        t = pq.read_table(p)
        n = t.num_rows

        state = _pack_columns(t, FLAT_STATE_COLS)
        action = _pack_columns(t, FLAT_ACTION_COLS)

        label_int = t.column("label").cast(pa.int64()).to_numpy(zero_copy_only=False)
        failure_label = pa.array((1 - label_int).astype(np.int64), type=pa.int64())

        for col in _FLAT_DROP_COLS:
            if col in t.column_names:
                t = t.drop([col])

        if "frame_index" in t.column_names:
            t = t.set_column(
                t.schema.get_field_index("frame_index"),
                "frame_index",
                t.column("frame_index").cast(pa.int64()),
            )

        t = t.append_column("observation.state", state)
        t = t.append_column("action", action)
        t = t.append_column("episode_index", pa.array(np.full(n, ep_idx, dtype=np.int64)))
        t = t.append_column(
            "index",
            pa.array(np.arange(frame_offset, frame_offset + n, dtype=np.int64)),
        )
        t = t.append_column("task_index", pa.array(np.zeros(n, dtype=np.int64)))
        t = t.append_column("failure_label", failure_label)

        tables.append(t)
        frame_offset += n

    combined = pa.concat_tables(tables)
    return Dataset(arrow_table=InMemoryTable(combined))


def flat_stats_to_array_stats(scalar_stats: dict) -> dict:
    """Pack scalar joint stats into per-feature length-6 array stats.

    normalize() in utils/processor_utils.py only touches keys present in
    POLICY_FEATURES and skips visual features, so we don't emit image stats
    or failure_label stats — they would be dead entries.
    """
    out: dict[str, dict] = {}
    for key, cols in [
        ("observation.state", FLAT_STATE_COLS),
        ("action", FLAT_ACTION_COLS),
    ]:
        if all(c in scalar_stats for c in cols):
            out[key] = {
                stat: np.array([scalar_stats[c][stat] for c in cols], dtype=np.float64)
                for stat in ("min", "max", "mean", "std")
            }
            out[key]["count"] = scalar_stats[cols[0]].get("count", 0)
    return out


def load_stats(local_dir: Path) -> dict | None:
    """Load meta/stats.json (scalar-keyed) and translate to array stats."""
    if not (local_dir / STATS_PATH).exists():
        return None
    raw = cast_stats_to_numpy(load_json(local_dir / STATS_PATH))
    return flat_stats_to_array_stats(raw)


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
