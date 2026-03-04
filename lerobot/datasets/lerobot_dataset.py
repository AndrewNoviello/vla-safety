import collections
import os
import time as _time
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
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_subtasks,
    load_tasks,
    resolve_delta_timestamps,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

# ---------------------------------------------------------------------------
# Per-worker LRU cache of open av.InputContainer objects.
# Module-level dicts are process-local, so each DataLoader worker has its own
# copy — no cross-worker state, no locks needed.
# ---------------------------------------------------------------------------
_VIDEO_CONTAINER_CACHE: dict[int, collections.OrderedDict] = {}
_VIDEO_CONTAINER_CACHE_MAX_SIZE: int = 4  # max open containers per worker

# Set LEROBOT_PROFILE_DECODE=1 to print per-window decode timing to stdout.
_PROFILE_DECODE: bool = os.environ.get("LEROBOT_PROFILE_DECODE", "0") == "1"


def _get_worker_id() -> int:
    """Return the current DataLoader worker id, or -1 in the main process."""
    info = torch.utils.data.get_worker_info()
    return info.id if info is not None else -1


def _get_cached_container(video_path: "Path"):
    """Return a cached open av.InputContainer for video_path (per-worker LRU).

    Opens a new container on cache miss; evicts the LRU entry when the cache
    exceeds _VIDEO_CONTAINER_CACHE_MAX_SIZE. Caller must NOT close the container.
    """
    import av
    worker_id = _get_worker_id()
    cache = _VIDEO_CONTAINER_CACHE.setdefault(worker_id, collections.OrderedDict())
    key = str(video_path)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    container = av.open(str(video_path))
    cache[key] = container
    cache.move_to_end(key)
    while len(cache) > _VIDEO_CONTAINER_CACHE_MAX_SIZE:
        _, old = cache.popitem(last=False)
        try:
            old.close()
        except Exception:
            pass
    return container


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

        self.root.mkdir(exist_ok=True, parents=True)
        revision = CODEBASE_VERSION

        try:
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            revision = get_safe_version(self.repo_id, revision)
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id, repo_type="dataset", revision=revision,
                local_dir=self.root, allow_patterns="meta/",
            )
            self._load_metadata()

        try:
            self.hf_dataset = self._load_hf_dataset()
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            revision = get_safe_version(self.repo_id, revision)
            files = self._get_episodes_file_paths() if self.episodes is not None else None
            snapshot_download(
                self.repo_id, repo_type="dataset", revision=revision,
                local_dir=self.root, allow_patterns=files,
            )
            self.hf_dataset = self._load_hf_dataset()

        video_keys = self._keys_by_dtype("video")
        for key in video_keys:
            video_dir = self.root / "videos" / key
            if not any(video_dir.rglob("*.mp4")):
                revision = get_safe_version(self.repo_id, revision)
                snapshot_download(
                    self.repo_id, repo_type="dataset", revision=revision,
                    local_dir=self.root, allow_patterns=f"videos/{key}/",
                )
                break

        self._absolute_to_relative_idx = (
            {int(idx): i for i, idx in enumerate(self.hf_dataset["index"])}
            if self.episodes is not None else None
        )

        if policy_type is not None and self.delta_timestamps is None:
            self.delta_timestamps = resolve_delta_timestamps(policy_type, self)
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.num_frames = len(self.hf_dataset) if self.episodes is not None else self._info["total_frames"]
        self.num_episodes = len(self.episodes) if self.episodes is not None else self._info["total_episodes"]

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_metadata(self):
        self._info = load_info(self.root)
        check_version_compatibility(
            self.repo_id,
            packaging.version.parse(self._info["codebase_version"]),
            CODEBASE_VERSION,
        )
        self._tasks = load_tasks(self.root)
        self._subtasks = load_subtasks(self.root)
        self._episodes = load_episodes(self.root)
        self._stats = load_stats(self.root)
        self.fps = self._info["fps"]
        self.features = self._info["features"]
        self.stats = self._stats

    def _keys_by_dtype(self, *dtypes: str) -> list[str]:
        return [key for key, ft in self._info["features"].items() if ft["dtype"] in dtypes]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_data_file_path(self, ep_index: int) -> Path:
        if ep_index >= len(self._episodes):
            raise IndexError(f"Episode index {ep_index} out of range ({len(self._episodes)} episodes)")
        ep = self._episodes[ep_index]
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

    @staticmethod
    def _decode_video_frames_window(
        video_path: "Path",
        timestamps: list[float],
        use_cache: bool = False,
    ) -> list[torch.Tensor]:
        """Decode multiple frames from one video file in a single open+seek+decode pass.

        Opens the container once, seeks to 0.1s before the earliest timestamp,
        then decodes forward collecting the best-match frame for each target.
        Returns tensors in the same order as the input timestamps list.

        Args:
            video_path: Path to the .mp4 file.
            timestamps: Target timestamps in seconds (need not be sorted).
            use_cache: If True, use the per-worker LRU container cache rather
                than opening/closing a fresh container. Only safe inside a
                DataLoader worker (persistent_workers=True recommended).
        """
        import av

        if not timestamps:
            return []

        # Sort timestamps but remember original positions to restore output order.
        order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        sorted_ts = [timestamps[i] for i in order]
        n = len(sorted_ts)

        t0 = _time.perf_counter() if _PROFILE_DECODE else 0.0

        # Open (or retrieve cached) container, then seek to just before first target.
        if use_cache:
            container = _get_cached_container(video_path)
        else:
            container = av.open(str(video_path))

        stream = container.streams.video[0]
        tb = float(stream.time_base)
        # Same 0.1s pre-seek margin as _decode_video_frame to land before a keyframe.
        seek_pts = int(max(0.0, sorted_ts[0] - 0.1) / tb)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        # Single forward decode pass: collect best-match frame for each target.
        # best[j] = [min_diff, best_av_frame] for sorted_ts[j]
        best: list[list] = [[float("inf"), None] for _ in range(n)]
        cursor = 0  # index into sorted_ts of the earliest unsettled target

        for frame in container.decode(stream):
            frame_ts = float(frame.pts * tb)
            for j in range(cursor, n):
                diff = abs(frame_ts - sorted_ts[j])
                if diff < best[j][0]:
                    best[j][0] = diff
                    best[j][1] = frame
            # Advance cursor past targets now more than 0.5s behind this frame
            # (same stop sentinel as _decode_video_frame).
            while cursor < n and frame_ts > sorted_ts[cursor] + 0.5:
                cursor += 1
            if cursor >= n:
                break

        if not use_cache:
            container.close()

        if _PROFILE_DECODE:
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            span = sorted_ts[-1] - sorted_ts[0] if n > 1 else 0.0
            print(
                f"[DECODE] worker={_get_worker_id()} file={video_path.name} "
                f"n={n} span={span:.3f}s elapsed={elapsed_ms:.1f}ms",
                flush=True,
            )

        # Convert frames to tensors, then restore the original (unsorted) order.
        sorted_tensors: list[torch.Tensor] = []
        for j in range(n):
            if best[j][1] is None:
                raise ValueError(
                    f"No frame found near timestamp {sorted_ts[j]:.4f}s in {video_path}"
                )
            img = best[j][1].to_ndarray(format="rgb24")  # (H, W, C) uint8
            sorted_tensors.append(
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W) float32
            )

        result: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
        for sorted_pos, orig_pos in enumerate(order):
            result[orig_pos] = sorted_tensors[sorted_pos]
        return result

    def _load_hf_dataset(self) -> datasets.Dataset:
        hf_features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=hf_features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        if len(self.hf_dataset) == 0:
            return False
        available = {int(ep) for ep in self.hf_dataset.unique("episode_index")}
        requested = set(range(self._info["total_episodes"])) if self.episodes is None else set(self.episodes)
        return requested.issubset(available)

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
        idx_map = self._absolute_to_relative_idx
        for key, q_idx in query_indices.items():
            rel = q_idx if idx_map is None else [idx_map[i] for i in q_idx]
            result[key] = torch.stack(self.hf_dataset[key][rel])
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"])
        abs_idx = int(item["index"])

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            # Video keys live in mp4 files, not parquet columns — skip them here.
            # The multi-frame video decode block below handles them via delta_indices.
            video_key_set = set(self._keys_by_dtype("video"))
            parquet_query_indices = {k: v for k, v in query_indices.items() if k not in video_key_set}
            query_result = self._query_hf_dataset(parquet_query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Decode video frames for video-dtype features — one open+seek per camera per sample.
        video_keys = self._keys_by_dtype("video")
        if video_keys:
            ep = self._episodes[ep_idx]
            base_timestamp = float(item["timestamp"])
            # Use per-worker container cache only inside a DataLoader worker process.
            _in_worker = torch.utils.data.get_worker_info() is not None
            for key in video_keys:
                from_ts = ep[f"videos/{key}/from_timestamp"]
                video_path = self._get_video_path(ep_idx, key)
                if self.delta_indices is not None and key in self.delta_indices:
                    timestamps = [
                        from_ts + base_timestamp + delta_idx / self.fps
                        for delta_idx in self.delta_indices[key]
                    ]
                else:
                    timestamps = [from_ts + base_timestamp]
                frames = self._decode_video_frames_window(
                    video_path, timestamps, use_cache=_in_worker
                )
                item[key] = torch.stack(frames) if len(frames) > 1 else frames[0]

        if self.image_transforms is not None:
            for cam in self._keys_by_dtype("image", "video"):
                item[cam] = self.image_transforms(item[cam])

        task_idx = int(item["task_index"])
        item["task"] = self._tasks.iloc[task_idx].name

        if "subtask_index" in self.features and self._subtasks is not None:
            subtask_idx = int(item["subtask_index"])
            item["subtask"] = self._subtasks.iloc[subtask_idx].name

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({{\n"
            f"    repo_id: '{self.repo_id}',\n"
            f"    num_episodes: {self.num_episodes},\n"
            f"    num_frames: {self.num_frames},\n"
            f"    features: {list(self.features)},\n"
            "}})"
        )
