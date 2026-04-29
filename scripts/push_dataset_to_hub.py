"""
Push a flat-format recording dataset folder to the Hugging Face Hub as a
`dataset` repo.

Expected folder layout (matches scripts/recorder.py output):

    <folder>/
      data/episode_NNN.parquet
      videos/episode_NNN.mp4
      meta/stats.json
      README.md

Usage:
    python scripts/push_dataset_to_hub.py <folder> <repo_id>

Auth: reads the cached token from `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ALLOW_PATTERNS = [
    "data/*.parquet",
    "videos/*.mp4",
    "meta/*.json",
    "README.md",
]


def push(folder: Path, repo_id: str) -> str:
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    if not (folder / "data").is_dir():
        raise SystemExit(f"{folder} has no data/ subdirectory")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    n_parquet = len(list((folder / "data").glob("episode_*.parquet")))
    n_mp4 = len(list((folder / "videos").glob("episode_*.mp4"))) if (folder / "videos").is_dir() else 0
    print(
        f"Uploading {folder} → {repo_id}\n"
        f"  parquets: {n_parquet}   videos: {n_mp4}",
        flush=True,
    )

    url = HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        allow_patterns=ALLOW_PATTERNS,
    )
    print(f"\nDone: {url}", flush=True)
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Path to the dataset folder to upload.")
    parser.add_argument("repo_id", help="Target repo id, e.g. 'owner/name'.")
    args = parser.parse_args(argv)

    push(args.folder, args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
