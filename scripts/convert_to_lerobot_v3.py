"""
Convert a flat-format recording dataset (the layout produced by
scripts/recorder.py / merge_recordings.py / filter_recordings.py) to the
LeRobot v3.0 dataset format **without re-encoding videos**.

Source layout (read-only — never modified by this script):

    <src>/
      data/episode_NNN.parquet
      videos/episode_NNN.mp4
      meta/stats.json
      README.md

Source per-frame columns:
    timestamp, frame_index,
    observation.state_j{1..6}, action_j{1..6},
    ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw,
    label  (1.0 = GOOD, 0.0 = BAD)

The ee_* and label columns are read from the source but **not** propagated to the
v3 output (kept minimal to match upstream LeRobot v3 datasets).

Target layout (LeRobot v3.0):

    <out>/
      data/chunk-000/file-000.parquet
      videos/observation.images.<cam>/chunk-000/file-000.mp4
      meta/info.json
      meta/stats.json
      meta/tasks.parquet
      meta/episodes/chunk-000/file-000.parquet

Conversion mapping:
    observation.state_j{1..6}      → observation.state          (float32, shape (6,))
    action_j{1..6}                 → action                     (float32, shape (6,))
    videos/episode_NNN.mp4         → observation.images.<cam>   (video, stream-copied, shape (H,W,3))

Video files are concatenated into v3 chunked files via PyAV's concat demuxer
with `add_stream_from_template(opaque=True)` — packet-level stream copy, no
decode + no re-encode.

Image stats are computed by sampling a small number of frames per episode
(default 30, configurable via --image-stats-samples).

Usage:
    python scripts/convert_to_lerobot_v3.py \\
        --src data/exp_success \\
        --out data/exp_success_v3 \\
        --task "pick up the middle domino"
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"^episode_(\d+)\.parquet$")
DEFAULT_FPS = 20
DEFAULT_IMAGE_SAMPLES = 30
DEFAULT_TASK_FALLBACK = "task"

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
STATE_COLS = [f"observation.state_{j}" for j in JOINT_NAMES]
ACTION_COLS = [f"action_{j}" for j in JOINT_NAMES]

QUANTILE_LEVELS = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = [f"q{int(round(q * 100)):02d}" for q in QUANTILE_LEVELS]


# ── Source helpers ────────────────────────────────────────────────────────


def list_episodes(src: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted((src / "data").glob("episode_*.parquet")):
        m = EPISODE_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def parse_task_from_readme(src: Path) -> str | None:
    readme = src / "README.md"
    if not readme.is_file():
        return None
    for line in readme.read_text().splitlines():
        line = line.strip()
        if line.lower().startswith("task:"):
            return line.split(":", 1)[1].strip() or None
    return None


def video_dims_from_first(episodes: list[tuple[int, Path]], src: Path) -> tuple[int, int]:
    for _, parquet in episodes:
        mp4 = src / "videos" / parquet.name.replace(".parquet", ".mp4")
        if not mp4.is_file():
            continue
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            cap.release()
            continue
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if h > 0 and w > 0:
            return h, w
    raise SystemExit(f"Could not read frame size from any video under {src}/videos")


def stack_cols(table: pa.Table, cols: list[str]) -> np.ndarray:
    return np.stack(
        [table.column(c).to_numpy().astype(np.float32) for c in cols],
        axis=1,
    )


# ── Stats ─────────────────────────────────────────────────────────────────


def vector_stats(arr: np.ndarray) -> dict:
    """arr shape (N, D). Per-feature stats with shape (D,)."""
    n = arr.shape[0]
    out = {
        "min": arr.min(axis=0).astype(np.float32),
        "max": arr.max(axis=0).astype(np.float32),
        "mean": arr.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": arr.std(axis=0, dtype=np.float64).astype(np.float32),
        "count": np.array([n], dtype=np.int64),
    }
    for q, qkey in zip(QUANTILE_LEVELS, QUANTILE_KEYS):
        out[qkey] = np.quantile(arr, q, axis=0).astype(np.float32)
    return out


def sample_video_frames(mp4_path: Path, n_samples: int) -> np.ndarray:
    """Return (N, 3, H, W) uint8 of evenly-spaced sampled frames."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open video for sampling: {mp4_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_take = min(n_samples, max(1, n_frames))
    if n_frames > 1:
        idxs = np.round(np.linspace(0, n_frames - 1, n_take)).astype(int).tolist()
    else:
        idxs = [0]

    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3)
        frames.append(rgb.transpose(2, 0, 1))  # → (3, H, W)
    cap.release()

    if not frames:
        raise SystemExit(f"No frames decoded from {mp4_path}")
    return np.stack(frames, axis=0)


def image_stats(frames_chw: np.ndarray) -> dict:
    """frames_chw shape (N, 3, H, W) uint8. Returns per-channel stats shape (3, 1, 1) in [0, 1]."""
    n = frames_chw.shape[0] * frames_chw.shape[2] * frames_chw.shape[3]
    f = frames_chw.astype(np.float32) / 255.0
    per_ch = f.transpose(1, 0, 2, 3).reshape(3, -1)  # (3, N*H*W)
    out = {
        "min": per_ch.min(axis=1).reshape(3, 1, 1).astype(np.float32),
        "max": per_ch.max(axis=1).reshape(3, 1, 1).astype(np.float32),
        "mean": per_ch.mean(axis=1).reshape(3, 1, 1).astype(np.float32),
        "std": per_ch.std(axis=1).reshape(3, 1, 1).astype(np.float32),
        "count": np.array([n], dtype=np.int64),
    }
    for q, qkey in zip(QUANTILE_LEVELS, QUANTILE_KEYS):
        out[qkey] = np.quantile(per_ch, q, axis=1).reshape(3, 1, 1).astype(np.float32)
    return out


# ── Per-frame data table ──────────────────────────────────────────────────


def build_data_table(
    state: np.ndarray,
    action: np.ndarray,
    episode_idx: int,
    global_offset: int,
    fps: int,
) -> pa.Table:
    n = state.shape[0]
    schema = pa.schema(
        [
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
            ("observation.state", pa.list_(pa.float32(), 6)),
            ("action", pa.list_(pa.float32(), 6)),
        ]
    )
    arrays = [
        pa.array((np.arange(n) / fps).astype(np.float32)),
        pa.array(np.arange(n, dtype=np.int64)),
        pa.array(np.full(n, episode_idx, dtype=np.int64)),
        pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64)),
        pa.array(np.zeros(n, dtype=np.int64)),
        pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1).astype(np.float32)), 6),
        pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1).astype(np.float32)), 6),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


# ── Main ──────────────────────────────────────────────────────────────────


def run(
    src: Path,
    out: Path,
    task: str,
    repo_id: str,
    camera_short: str,
    fps: int,
    robot_type: str | None,
    image_stats_samples: int,
    force: bool,
) -> None:
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")
    if not (src / "data").is_dir():
        raise SystemExit(f"{src} has no data/ subdirectory")
    if not (src / "videos").is_dir():
        raise SystemExit(f"{src} has no videos/ subdirectory (v3 needs frames)")

    if out.exists():
        if any(out.iterdir()) and not force:
            raise SystemExit(
                f"Output {out} exists and is non-empty. Re-run with --force to overwrite."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = list_episodes(src)
    if not episodes:
        raise SystemExit(f"No episode_*.parquet files found in {src}/data")

    height, width = video_dims_from_first(episodes, src)
    camera_key = f"observation.images.{camera_short}"

    print(
        f"Converting {len(episodes)} episodes from {src}\n"
        f"  → {out}\n"
        f"  fps={fps}  resolution={width}x{height}  camera_key={camera_key}\n"
        f"  task={task!r}  repo_id={repo_id!r}\n"
        f"  image_stats_samples={image_stats_samples}",
        flush=True,
    )

    # Imports deferred so --help works without lerobot installed.
    from datasets import Dataset
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_PATH,
        create_empty_dataset_info,
        flatten_dict,
        write_episodes,
        write_info,
        write_stats,
        write_tasks,
    )
    from lerobot.datasets.video_utils import (
        concatenate_video_files,
        get_video_duration_in_s,
        get_video_info,
    )

    # ── Pass 1: per-episode work ──────────────────────────────────────────
    per_ep_tables: list[pa.Table] = []
    per_ep_stats: list[dict] = []
    per_ep_meta: list[dict] = []
    src_mp4s: list[Path] = []
    global_offset = 0
    cum_duration_s = 0.0

    for ep_idx, (orig_idx, parquet) in enumerate(episodes):
        table = pq.read_table(parquet)
        ep_len = table.num_rows
        if ep_len == 0:
            print(f"  ⚠ episode {orig_idx:03d}: 0 rows, skipping", flush=True)
            continue

        mp4 = src / "videos" / parquet.name.replace(".parquet", ".mp4")
        if not mp4.is_file():
            raise SystemExit(f"Missing video for {parquet.name}: {mp4}")

        state = stack_cols(table, STATE_COLS)
        action = stack_cols(table, ACTION_COLS)

        ep_duration_s = float(get_video_duration_in_s(mp4))
        sampled = sample_video_frames(mp4, image_stats_samples)

        ep_stats = {
            "observation.state": vector_stats(state),
            "action": vector_stats(action),
            camera_key: image_stats(sampled),
        }
        per_ep_stats.append(ep_stats)

        per_ep_tables.append(
            build_data_table(state, action, ep_idx, global_offset, fps)
        )

        per_ep_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task],
                "length": ep_len,
                "dataset_from_index": global_offset,
                "dataset_to_index": global_offset + ep_len,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{camera_key}/chunk_index": 0,
                f"videos/{camera_key}/file_index": 0,
                f"videos/{camera_key}/from_timestamp": cum_duration_s,
                f"videos/{camera_key}/to_timestamp": cum_duration_s + ep_duration_s,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )

        src_mp4s.append(mp4)
        global_offset += ep_len
        cum_duration_s += ep_duration_s

        if (ep_idx + 1) % 20 == 0 or ep_idx + 1 == len(episodes):
            print(
                f"  scanned {ep_idx + 1}/{len(episodes)} episodes "
                f"({global_offset} frames, {cum_duration_s:.1f}s)",
                flush=True,
            )

    total_frames = global_offset
    total_episodes = len(per_ep_meta)

    # ── Pass 2: write data parquet ───────────────────────────────────────
    data_path = out / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(per_ep_tables), data_path, compression="snappy")
    print(f"  ✓ wrote {data_path}", flush=True)

    # ── Pass 2: concat videos (stream copy) ──────────────────────────────
    video_path = out / DEFAULT_VIDEO_PATH.format(
        video_key=camera_key, chunk_index=0, file_index=0
    )
    concatenate_video_files(src_mp4s, video_path)
    print(f"  ✓ stream-copied {len(src_mp4s)} videos → {video_path}", flush=True)

    # ── Build features dict (with video info from concatenated file) ─────
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
            "fps": fps,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
            "fps": fps,
        },
        camera_key: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
            "info": get_video_info(video_path),
        },
        "timestamp": {"dtype": "float32", "shape": (1,)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
    }

    # ── meta/episodes parquet ────────────────────────────────────────────
    ep_rows = []
    for meta_row, stats_row in zip(per_ep_meta, per_ep_stats):
        row = dict(meta_row)
        row.update(flatten_dict({"stats": stats_row}))
        ep_rows.append(row)

    # Use Dataset.from_list rather than from_pandas: pandas→pyarrow chokes on
    # ndarray cells with ndim > 1 (image stats are shape (3,1,1)). HF Datasets'
    # own type inference handles nested arrays via Sequence(Sequence(...)).
    ep_dataset = Dataset.from_list(ep_rows)
    write_episodes(ep_dataset, out)
    print(f"  ✓ wrote {out}/meta/episodes/chunk-000/file-000.parquet", flush=True)

    # ── meta/tasks.parquet ───────────────────────────────────────────────
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task])
    write_tasks(tasks_df, out)
    print(f"  ✓ wrote {out}/meta/tasks.parquet", flush=True)

    # ── meta/stats.json (global aggregate) ───────────────────────────────
    global_stats = aggregate_stats(per_ep_stats)
    write_stats(global_stats, out)
    print(f"  ✓ wrote {out}/meta/stats.json", flush=True)

    # ── meta/info.json ───────────────────────────────────────────────────
    info = create_empty_dataset_info(
        codebase_version="v3.0",
        fps=fps,
        features=features,
        use_videos=True,
        robot_type=robot_type,
        chunks_size=DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    )
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, out)
    print(f"  ✓ wrote {out}/meta/info.json", flush=True)

    print(
        f"\nDone. {total_episodes} episodes, {total_frames} frames "
        f"({cum_duration_s:.1f}s of video) → {out}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True, type=Path, help="Source flat-format dataset directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the v3 dataset.")
    parser.add_argument(
        "--task",
        default=None,
        help="Natural-language task string. Defaults to the 'Task: ...' line in <src>/README.md.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Identifier stored inside info.json. Defaults to 'local/<src.name>'. Not pushed.",
    )
    parser.add_argument(
        "--camera-key",
        default="front",
        help="Short camera name; produces 'observation.images.<camera-key>'. Default: front.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Default: {DEFAULT_FPS}.")
    parser.add_argument(
        "--robot-type",
        default=None,
        help="Optional robot_type string written to info.json (e.g. 'so101').",
    )
    parser.add_argument(
        "--image-stats-samples",
        type=int,
        default=DEFAULT_IMAGE_SAMPLES,
        help=f"Frames sampled per episode for image-feature stats. Default: {DEFAULT_IMAGE_SAMPLES}.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty --out.")
    args = parser.parse_args(argv)

    task = args.task or parse_task_from_readme(args.src) or DEFAULT_TASK_FALLBACK
    repo_id = args.repo_id or f"local/{args.src.name}"

    run(
        src=args.src,
        out=args.out,
        task=task,
        repo_id=repo_id,
        camera_short=args.camera_key,
        fps=args.fps,
        robot_type=args.robot_type,
        image_stats_samples=args.image_stats_samples,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
