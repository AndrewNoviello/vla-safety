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
from collections.abc import Callable
from pathlib import Path

import datasets
import packaging.version
import torch
import torch.utils
from huggingface_hub import snapshot_download

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.transforms import ImageTransformsConfig, ImageTransforms
from lerobot.datasets.utils import (
    check_delta_timestamps,
    check_version_compatibility,
    get_delta_indices,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_subtasks,
    load_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

CODEBASE_VERSION = "v3.0"


class LeRobotDataset(torch.utils.data.Dataset):
    """Load-only dataset for VLA training and inference.

    Loads image-based datasets from HuggingFace Hub or a local directory.
    Recording, video encoding/decoding, and streaming are not supported.
    """

    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 0.04,
        root: Path | str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.delta_indices = None

        revision = CODEBASE_VERSION
        self.root.mkdir(exist_ok=True, parents=True)

        # --- load metadata (inlined from the former LeRobotDatasetMetadata) ---
        try:
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(revision):
                revision = get_safe_version(self.repo_id, revision)
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=self.root,
                allow_patterns="meta/",
            )
            self._load_metadata()

        # --- download data files if needed ---
        try:
            self.hf_dataset = self._load_hf_dataset()
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            if is_valid_version(revision):
                revision = get_safe_version(self.repo_id, revision)
            files = self._get_episodes_file_paths() if self.episodes is not None else None
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=self.root,
                allow_patterns=files,
            )
            self.hf_dataset = self._load_hf_dataset()

        # absolute -> relative index mapping for episode subsets
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_metadata(self):
        self._info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self._tasks = load_tasks(self.root)
        self._subtasks = load_subtasks(self.root)
        self._episodes = load_episodes(self.root)
        self._stats = load_stats(self.root)

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(self._info["codebase_version"])

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        return self._info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self._info["features"]

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["image", "video"]]

    @property
    def shapes(self) -> dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def num_frames(self) -> int:
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._info["total_frames"]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self._info["total_episodes"]

    @property
    def total_episodes(self) -> int:
        return self._info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self._info["total_frames"]

    # ------------------------------------------------------------------
    # Backward-compat shim: code that reads dataset.meta.*
    # ------------------------------------------------------------------

    class _MetaProxy:
        """Thin proxy so that ``dataset.meta.stats``, ``dataset.meta.fps``, etc. still work."""

        def __init__(self, ds: "LeRobotDataset"):
            self._ds = ds

        @property
        def stats(self):
            return self._ds._stats

        @stats.setter
        def stats(self, value):
            self._ds._stats = value

        @property
        def fps(self):
            return self._ds.fps

        @property
        def info(self):
            return self._ds._info

        @property
        def features(self):
            return self._ds.features

        @property
        def camera_keys(self):
            return self._ds.camera_keys

        @property
        def tasks(self):
            return self._ds._tasks

        @property
        def subtasks(self):
            return self._ds._subtasks

        @property
        def episodes(self):
            return self._ds._episodes

        @property
        def total_episodes(self):
            return self._ds.total_episodes

        @property
        def total_frames(self):
            return self._ds.total_frames

        @property
        def image_keys(self):
            return self._ds.image_keys

        @property
        def video_keys(self):
            return [key for key, ft in self._ds.features.items() if ft["dtype"] == "video"]

        @property
        def data_path(self) -> str:
            return self._ds._info["data_path"]

        def get_data_file_path(self, ep_index: int) -> Path:
            return self._ds._get_data_file_path(ep_index)

    @property
    def meta(self) -> "_MetaProxy":
        if not hasattr(self, "_meta_proxy"):
            self._meta_proxy = self._MetaProxy(self)
        return self._meta_proxy

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_data_file_path(self, ep_index: int) -> Path:
        eps = self._episodes
        if eps is None:
            eps = load_episodes(self.root)
        if ep_index >= len(eps):
            raise IndexError(f"Episode index {ep_index} out of range ({len(eps)} episodes)")
        ep = eps[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self._info["data_path"].format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def _get_episodes_file_paths(self) -> list[str]:
        episodes = self.episodes if self.episodes is not None else list(range(self._info["total_episodes"]))
        fpaths = list({str(self._get_data_file_path(ep_idx)) for ep_idx in episodes})
        return fpaths

    def _load_hf_dataset(self) -> datasets.Dataset:
        hf_features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=hf_features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False
        available = {
            ep.item() if isinstance(ep, torch.Tensor) else ep
            for ep in self.hf_dataset.unique("episode_index")
        }
        requested = set(range(self._info["total_episodes"])) if self.episodes is None else set(self.episodes)
        return requested.issubset(available)

    @property
    def hf_features(self) -> datasets.Features:
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        return get_hf_features_from_features(self.features)

    # ------------------------------------------------------------------
    # Delta-timestamp query helpers
    # ------------------------------------------------------------------

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep = self._episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self._tasks.iloc[task_idx].name

        if "subtask_index" in self.features and self._subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._subtasks.iloc[subtask_idx].name

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{list(self.features)}',\n"
            "})',\n"
        )


# ------------------------------------------------------------------
# Functional API
# ------------------------------------------------------------------


def load_dataset(
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
    image_transforms: ImageTransformsConfig | Callable | None = None,
    root: Path | str | None = None,
    tolerance_s: float = 0.04,
) -> LeRobotDataset:
    """One-liner to load a dataset from HuggingFace Hub or local disk."""
    if isinstance(image_transforms, ImageTransformsConfig):
        image_transforms = ImageTransforms(image_transforms) if image_transforms.enable else None
    return LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        tolerance_s=tolerance_s,
        root=root,
    )


# ------------------------------------------------------------------
# Multi-dataset wrapper
# ------------------------------------------------------------------


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying ``LeRobotDataset`` instances.

    The underlying datasets are effectively concatenated.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
            )
            for repo_id in repo_ids
        ]

        self.disabled_features: set[str] = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained "
                    "in all the other datasets."
                )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats([ds.stats for ds in self._datasets])

    @property
    def fps(self) -> int:
        return self._datasets[0].fps

    @property
    def features(self) -> dict:
        feats: dict = {}
        for ds in self._datasets:
            feats.update({k: v for k, v in ds.features.items() if k not in self.disabled_features})
        return feats

    @property
    def num_frames(self) -> int:
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(d.num_episodes for d in self._datasets)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("Index within bounds but loop did not break.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f")"
        )
