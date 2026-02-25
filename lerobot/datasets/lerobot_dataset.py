import logging
from collections.abc import Callable
from pathlib import Path

import datasets
import packaging.version
import torch
import torch.utils
from huggingface_hub import snapshot_download

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
    resolve_delta_timestamps,
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
        policy_type: str | None = None,
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

        # --- download video files if needed ---
        video_keys = [key for key, ft in self.features.items() if ft["dtype"] == "video"]
        if video_keys:
            for key in video_keys:
                video_dir = self.root / "videos" / key
                if not any(video_dir.rglob("*.mp4")):
                    if is_valid_version(revision):
                        revision = get_safe_version(self.repo_id, revision)
                    snapshot_download(
                        self.repo_id,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=self.root,
                        allow_patterns=f"videos/{key}/",
                    )
                    break  # single download covers all video keys

        # absolute -> relative index mapping for episode subsets
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

        if policy_type is not None and self.delta_timestamps is None:
            self.delta_timestamps = resolve_delta_timestamps(policy_type, self)

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
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["image", "video"]]

    @property
    def shapes(self) -> dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def num_frames(self) -> int:
        if self.episodes is not None:
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

    @property
    def meta(self) -> "LeRobotDataset":
        """Metadata interface compatible with make_policy (dataset has .features, .stats, etc.)."""
        return self

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

    def _get_video_path(self, ep_idx: int, key: str) -> Path:
        ep = self._episodes[ep_idx]
        chunk_idx = ep[f"videos/{key}/chunk_index"]
        file_idx = ep[f"videos/{key}/file_index"]
        video_path_template = self._info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        path = video_path_template.format(video_key=key, chunk_index=chunk_idx, file_index=file_idx)
        return self.root / path

    @staticmethod
    def _decode_video_frame(video_path: Path, timestamp: float) -> torch.Tensor:
        """Decode the frame nearest to `timestamp` (seconds) from an mp4 file."""
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            seek_pts = int(max(0.0, timestamp - 0.1) / tb)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            best_frame = None
            best_diff = float("inf")
            for frame in container.decode(stream):
                frame_ts = float(frame.pts * tb)
                diff = abs(frame_ts - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = frame
                if frame_ts > timestamp + 0.5:
                    break
        if best_frame is None:
            raise ValueError(f"Could not find frame at timestamp {timestamp} in {video_path}")
        img = best_frame.to_ndarray(format="rgb24")  # (H, W, C) uint8
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W) float [0,1]

    def _load_hf_dataset(self) -> datasets.Dataset:
        hf_features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=hf_features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        if len(self.hf_dataset) == 0:
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
            result[key] = torch.stack(self.hf_dataset[key][relative_indices])
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

        # Decode video frames for video-dtype features
        video_keys = [key for key, ft in self.features.items() if ft["dtype"] == "video"]
        if video_keys:
            ep = self._episodes[ep_idx]
            timestamp = item["timestamp"].item()
            for key in video_keys:
                from_ts = ep[f"videos/{key}/from_timestamp"]
                abs_timestamp = from_ts + timestamp
                video_path = self._get_video_path(ep_idx, key)
                item[key] = self._decode_video_frame(video_path, abs_timestamp)

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
    image_transforms: Callable | None = None,
    root: Path | str | None = None,
    tolerance_s: float = 0.04,
    policy_type: str | None = None,
) -> LeRobotDataset:
    """One-liner to load a dataset from HuggingFace Hub or local disk.

    When policy_type is provided and delta_timestamps is not, delta_timestamps are
    resolved automatically from the policy's configuration (e.g. PI0 uses 50-step
    action chunks).
    """
    return LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        tolerance_s=tolerance_s,
        root=root,
        policy_type=policy_type,
    )
