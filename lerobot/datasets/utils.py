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
import contextlib
import importlib.resources
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from pprint import pformat
from typing import Any

import datasets
import numpy as np
import packaging.version
import pandas
import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_PREFIX, OBS_STR, REWARD
from lerobot.utils.utils import is_valid_numpy_dtype_string, suppress_progress_bars

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_SUBTASKS_PATH = "meta/subtasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


# ---------------------------------------------------------------------------
# Version compatibility (inlined from backward_compatibility.py)
# ---------------------------------------------------------------------------

V30_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v3.0 which is not backward compatible with v2.1.
Please, update your dataset to the new format using this command:
```
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

If you already have a converted version uploaded to the hub, then this error might be because of
an older version in your local cache. Consider deleting the cached version and retrying.

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError(
                "Contact the maintainer on [Discord](https://discord.com/invite/s3KuuzsPFb)."
            )
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Parquet / file helpers
# ---------------------------------------------------------------------------


def get_parquet_file_size_in_mb(parquet_path: str | Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            col_metadata = rg_metadata.column(column)
            total_uncompressed_size += col_metadata.total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def get_hf_dataset_size_in_mb(hf_ds: Dataset) -> int:
    return hf_ds.data.nbytes // (1024**2)


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def load_nested_dataset(
    pq_dir: Path, features: datasets.Features | None = None, episodes: list[int] | None = None
) -> Dataset:
    """Find parquet files in *pq_dir*/chunk-xxx/file-xxx.parquet and return an HF Dataset."""
    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")
    with suppress_progress_bars():
        filters = pa_ds.field("episode_index").isin(episodes) if episodes is not None else None
        return Dataset.from_parquet([str(path) for path in paths], filters=filters, features=features)


def get_parquet_num_frames(parquet_path: str | Path) -> int:
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def get_file_size_in_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024**2)


# ---------------------------------------------------------------------------
# Dict serialization helpers
# ---------------------------------------------------------------------------


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = outdict
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor | np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int | float | list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int | float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    fmt = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**fmt)
    return dataset


# ---------------------------------------------------------------------------
# JSON / metadata I/O
# ---------------------------------------------------------------------------


def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_info(info: dict, local_dir: Path) -> None:
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path) -> None:
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_tasks(tasks: pandas.DataFrame, local_dir: Path) -> None:
    path = local_dir / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def load_tasks(local_dir: Path) -> pandas.DataFrame:
    return pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)


def load_subtasks(local_dir: Path) -> pandas.DataFrame | None:
    subtasks_path = local_dir / DEFAULT_SUBTASKS_PATH
    if subtasks_path.exists():
        return pd.read_parquet(subtasks_path)
    return None


def write_episodes(episodes: Dataset, local_dir: Path) -> None:
    episode_size_mb = get_hf_dataset_size_in_mb(episodes)
    if episode_size_mb > DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise NotImplementedError(
            f"Episodes dataset is too large ({episode_size_mb} MB) to write to a single file."
        )
    fpath = local_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(fpath)


def load_episodes(local_dir: Path) -> datasets.Dataset:
    episodes = load_nested_dataset(local_dir / EPISODES_DIR)
    episodes = episodes.select_columns([key for key in episodes.features if not key.startswith("stats/")])
    return episodes


# ---------------------------------------------------------------------------
# HF transform
# ---------------------------------------------------------------------------


def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:
    """Convert a batch from a Hugging Face dataset to torch tensors."""
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def is_valid_version(version: str) -> bool:
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))
    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"Your dataset must be tagged with a codebase version. "
            f"See docs for tagging '{repo_id}'."
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


# ---------------------------------------------------------------------------
# HF features conversion
# ---------------------------------------------------------------------------


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")
    return datasets.Features(hf_features)


def _validate_feature_names(features: dict[str, dict]) -> None:
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


# ---------------------------------------------------------------------------
# Feature conversion helpers
# ---------------------------------------------------------------------------


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy features using fixed key naming conventions."""
    result: dict[str, PolicyFeature] = {}
    for key, feat in features.items():
        shape = feat["shape"]
        if feat["dtype"] in ["image", "video"]:
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = feat["names"]
            if names[2] in ["channel", "channels"]:
                shape = (shape[2], shape[0], shape[1])
            result[key] = PolicyFeature(FeatureType.VISUAL, shape)
        elif key == "observation.state":
            result[key] = PolicyFeature(FeatureType.STATE, shape)
        elif key == OBS_ENV_STATE:
            result[key] = PolicyFeature(FeatureType.ENV, shape)
        elif key.startswith("observation.language"):
            result[key] = PolicyFeature(FeatureType.LANGUAGE, shape)
        elif key.startswith(ACTION):
            result[key] = PolicyFeature(FeatureType.ACTION, shape)
        elif key.startswith(OBS_STR):
            result[key] = PolicyFeature(FeatureType.STATE, shape)
    return result


def combine_feature_dicts(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue
            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )
            if is_vector:
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                target["shape"] = (len(target["names"]),)
            else:
                out[key] = value
    return out


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": chunks_size or DEFAULT_CHUNK_SIZE,
        "data_files_size_in_mb": data_files_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": video_files_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Delta timestamps
# ---------------------------------------------------------------------------


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]
    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"The following delta_timestamps are found outside of tolerance range.\n"
                f"Please make sure they are multiples of 1/{fps} +/- tolerance.\n"
                f"{pformat(outside_tolerance)}"
            )
        return False
    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]
    return delta_indices


def resolve_delta_timestamps(
    policy_type: str, ds_meta
) -> dict[str, list] | None:
    """Resolve delta_timestamps from policy type. For PI0: action_delta_indices = list(range(50))."""
    delta_timestamps: dict[str, list] = {}
    features = ds_meta.features if hasattr(ds_meta, "features") else ds_meta._info["features"]
    fps = ds_meta.fps if hasattr(ds_meta, "fps") else ds_meta._info["fps"]

    # PI0: chunk_size=50, action_delta_indices = [0..49]
    if policy_type == "pi0":
        action_delta_indices = list(range(50))
        for key in features:
            if key == ACTION:
                delta_timestamps[key] = [i / fps for i in action_delta_indices]
        return delta_timestamps if delta_timestamps else None

    # Other policies: use PreTrainedConfig
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.get_choice_class(policy_type)()
    for key in features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]
    return delta_timestamps if delta_timestamps else None


# ---------------------------------------------------------------------------
# Dataloader helper
# ---------------------------------------------------------------------------


def cycle(iterable: Any) -> Iterator[Any]:
    """Dataloader-safe cyclical iterator (unlike itertools.cycle, works with multi-worker DataLoader)."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:
    api = HfApi()
    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)
    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    card_tags = ["LeRobot"]
    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )
    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()
    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_frame(frame: dict, features: dict) -> None:
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)
    if "task" not in actual_features:
        raise ValueError("Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n")
    actual_features_for_validation = actual_features - {"task"}
    error_message = validate_features_presence(actual_features_for_validation, expected_features)
    common_features = actual_features_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])
    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features
    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"
    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape
        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"
        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"
    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"
    return error_message


def validate_feature_string(name: str, value: str) -> str:
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")
    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")
    if episode_buffer["episode_index"] != total_episodes:
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset."
        )
    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")
    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )


def to_parquet_with_hf_images(
    df: pandas.DataFrame, path: Path, features: datasets.Features | None = None
) -> None:
    ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=features)
    ds.to_parquet(path)


def item_to_torch(item: dict) -> dict:
    for key, val in item.items():
        if isinstance(val, (np.ndarray | list)) and key not in ["task"]:
            item[key] = torch.tensor(val)
    return item
