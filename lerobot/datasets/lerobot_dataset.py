from collections.abc import Callable
from pathlib import Path

import datasets
import torch
from huggingface_hub import snapshot_download

from lerobot.datasets.utils import (
    PARQUET_FEATURES,
    POLICY_FEATURES,
    hf_transform_to_torch,
    load_episode_parquets,
    load_stats,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        delta_indices: dict[str, list[int]] | None = None,
        image_transforms: Callable | None = None,
        prompt: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_indices = delta_indices
        self.prompt = ""
        self.root.mkdir(exist_ok=True, parents=True)
        self._video_dir = self.root / "videos"
        self._decoder_cache: dict = {}

        meta_ok = (self.root / "meta" / "stats.json").exists()
        data_ok = any((self.root / "data").glob("episode_*.parquet"))
        video_ok = any(self._video_dir.glob("episode_*.mp4"))
        if not meta_ok or not data_ok or not video_ok:
            snapshot_download(
                self.repo_id, repo_type="dataset", revision="main",
                local_dir=self.root, allow_patterns=["meta/*", "data/*", "videos/*"],
            )

        self._load_metadata()
        self.hf_dataset = self._load_hf_dataset()

        self._episode_boundaries = self._build_episode_boundaries()

        self.num_frames = len(self.hf_dataset)
        self.num_episodes = len(set(self.hf_dataset["episode_index"]))

    def _load_metadata(self):
        self._stats = load_stats(self.root)
        self.features = POLICY_FEATURES
        self.stats = self._stats

    def _keys_by_dtype(self, *dtypes: str) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in dtypes]

    def _build_episode_boundaries(self) -> dict[int, tuple[int, int]]:
        episode_indices = self.hf_dataset["episode_index"]
        boundaries: dict[int, tuple[int, int]] = {}
        start = 0
        for i in range(len(episode_indices)):
            if i > 0 and episode_indices[i] != episode_indices[i - 1]:
                ep = int(episode_indices[i - 1])
                boundaries[ep] = (start, i)
                start = i
        if len(episode_indices) > 0:
            ep = int(episode_indices[-1])
            boundaries[ep] = (start, len(episode_indices))
        return boundaries

    def _get_video_decoder(self, video_path: Path):
        key = str(video_path)
        if key not in self._decoder_cache:
            from torchcodec.decoders import VideoDecoder
            self._decoder_cache[key] = VideoDecoder(key)
        return self._decoder_cache[key]

    def _load_video_frames(self, ep_idx: int, frame_indices: list[int]) -> torch.Tensor:
        path = self._video_dir / f"episode_{ep_idx:03d}.mp4"
        decoder = self._get_video_decoder(path)
        batch = decoder.get_frames_at(indices=frame_indices)
        return (batch.data / 255.0).float()

    def _load_hf_dataset(self) -> datasets.Dataset:
        hf_dataset = load_episode_parquets(self.root / "data", features=PARQUET_FEATURES)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start, ep_end = self._episode_boundaries.get(ep_idx, (0, len(self.hf_dataset)))
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

    def _query_hf_dataset(self, query_indices: dict[str, list[int]], ep_idx: int | None = None) -> dict:
        result: dict = {}
        video_keys = set(self._keys_by_dtype("image", "video"))
        for key, q_idx in query_indices.items():
            if key in video_keys:
                frame_indices = [int(self.hf_dataset[i]["frame_index"]) for i in q_idx]
                result[key] = self._load_video_frames(ep_idx, frame_indices)
            else:
                result[key] = torch.stack(self.hf_dataset[key][q_idx])
        return result

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"])
        abs_idx = int(item["index"])
        frame_idx = int(item["frame_index"])

        item["image"] = self._load_video_frames(ep_idx, [frame_idx]).squeeze(0)


        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices, ep_idx)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.image_transforms is not None:
            for cam in self._keys_by_dtype("image"):
                item[cam] = self.image_transforms(item[cam])

        task_idx = int(item["task_index"])
        item["task"] = self.prompt

        return item