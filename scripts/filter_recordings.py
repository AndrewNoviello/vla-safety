"""
Pick episodes from a flat-format recording dataset by predicate, copy the
selected ones into a new dataset directory, and recompute stats over only
the selected frames.

Source layout (same as scripts/recorder.py output):

    <source>/
      data/episode_NNN.parquet
      videos/episode_NNN.mp4
      meta/stats.json
      README.md

Output layout: identical, with episodes renumbered sequentially starting at 0.

The source can be a local directory or a Hugging Face dataset repo id
(downloaded via `snapshot_download` into a local cache before filtering).

Available filters:
    no-failures   — keep episodes whose `label` column is 1.0 for every frame
    all           — keep every episode

Usage:
    python scripts/filter_recordings.py \\
        --out data/exp_success \\
        --filter no-failures \\
        hf:various-and-sundry/domino-trajectories-on-new-table
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


HF_PREFIX = "hf:"
EPISODE_RE = re.compile(r"^episode_(\d+)\.parquet$")
ALLOW_PATTERNS = ["data/*.parquet", "videos/*.mp4", "meta/*.json", "README.md"]

# Same scalar columns tracked by RunningStats in scripts/recorder.py.
SCALAR_COLS = (
    [f"observation.state_j{i+1}" for i in range(6)]
    + [f"action_j{i+1}" for i in range(6)]
    + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    + ["label"]
)


# ── Source resolution ─────────────────────────────────────────────────────


def _looks_like_hf_id(s: str) -> bool:
    if s.startswith(HF_PREFIX):
        return True
    if Path(s).exists():
        return False
    return "/" in s and not s.startswith((".", "/"))


def resolve_source(source: str, hf_cache: Path) -> Path:
    if not _looks_like_hf_id(source):
        path = Path(source)
        if not path.is_dir():
            raise FileNotFoundError(f"Source not found: {source}")
        return path

    repo_id = source[len(HF_PREFIX):] if source.startswith(HF_PREFIX) else source
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            f"huggingface_hub is required to download '{source}'. "
            f"Install it with: pip install huggingface_hub"
        ) from e

    sanitized = repo_id.replace("/", "__")
    local_dir = hf_cache / sanitized
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ downloading {repo_id} → {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    return local_dir


def list_episodes(src: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted((src / "data").glob("episode_*.parquet")):
        m = EPISODE_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


# ── Predicates ────────────────────────────────────────────────────────────


def _no_failures(parquet: Path) -> bool:
    labels = pq.read_table(parquet, columns=["label"]).column("label").to_numpy()
    return bool((labels == 1.0).all())


def _all(_parquet: Path) -> bool:
    return True


PREDICATES = {
    "no-failures": _no_failures,
    "all": _all,
}


# ── Stats accumulator (matches meta/stats.json schema) ────────────────────


class StatsAccumulator:
    """Welford running stats over the scalar columns, fed parquet batches."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.n: dict[str, int] = {c: 0 for c in cols}
        self.mean: dict[str, float] = {c: 0.0 for c in cols}
        self.M2: dict[str, float] = {c: 0.0 for c in cols}
        self.min: dict[str, float] = {c: float("inf") for c in cols}
        self.max: dict[str, float] = {c: float("-inf") for c in cols}

    def update_table(self, table: pq.lib.Table) -> None:
        names = set(table.schema.names)
        for col in self.cols:
            if col not in names:
                continue
            arr = table.column(col).to_numpy()
            n_b = arr.size
            if n_b == 0:
                continue
            mean_b = float(arr.mean())
            M2_b = float(((arr - mean_b) ** 2).sum())
            min_b = float(arr.min())
            max_b = float(arr.max())

            n_a = self.n[col]
            if n_a == 0:
                self.n[col] = n_b
                self.mean[col] = mean_b
                self.M2[col] = M2_b
                self.min[col] = min_b
                self.max[col] = max_b
            else:
                n = n_a + n_b
                delta = mean_b - self.mean[col]
                self.mean[col] += delta * n_b / n
                self.M2[col] += M2_b + delta * delta * n_a * n_b / n
                self.n[col] = n
                if min_b < self.min[col]:
                    self.min[col] = min_b
                if max_b > self.max[col]:
                    self.max[col] = max_b

    def to_dict(self) -> dict:
        out: dict = {}
        for c in self.cols:
            n = self.n[c]
            std = math.sqrt(self.M2[c] / n) if n > 0 else 0.0
            out[c] = {
                "min": self.min[c] if n else None,
                "max": self.max[c] if n else None,
                "mean": self.mean[c] if n else None,
                "std": std if n else None,
                "count": n,
            }
        return out


# ── Main ──────────────────────────────────────────────────────────────────


def run(source: str, out: Path, hf_cache: Path, force: bool, predicate_name: str) -> None:
    if predicate_name not in PREDICATES:
        raise SystemExit(f"Unknown filter: {predicate_name}. Choices: {list(PREDICATES)}")
    predicate = PREDICATES[predicate_name]

    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(
                f"Output directory {out} already exists and is non-empty. "
                f"Re-run with --force to overwrite."
            )
        shutil.rmtree(out)

    out_data = out / "data"
    out_videos = out / "videos"
    out_meta = out / "meta"
    for d in (out_data, out_videos, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    src = resolve_source(source, hf_cache)
    if not (src / "data").is_dir():
        raise SystemExit(f"{src} has no data/ subdirectory")
    if not (src / "videos").is_dir():
        print(f"  ⚠ {src} has no videos/ subdirectory — videos will be skipped", flush=True)

    episodes = list_episodes(src)
    print(f"Scanning {len(episodes)} episodes from {source} (filter: {predicate_name})", flush=True)

    stats = StatsAccumulator(SCALAR_COLS)
    selected = 0
    skipped: list[int] = []

    for orig_idx, parquet in episodes:
        if not predicate(parquet):
            skipped.append(orig_idx)
            continue

        dst_name = f"episode_{selected:03d}"
        shutil.copy2(parquet, out_data / f"{dst_name}.parquet")

        mp4 = src / "videos" / parquet.name.replace(".parquet", ".mp4")
        if mp4.is_file():
            shutil.copy2(mp4, out_videos / f"{dst_name}.mp4")
        else:
            print(f"  ⚠ no video for {parquet.name}", flush=True)

        # Read again (full table) to update stats only over selected episodes.
        stats.update_table(pq.read_table(parquet))
        selected += 1

    (out_meta / "stats.json").write_text(json.dumps(stats.to_dict(), indent=2))

    cmd = (
        f"python scripts/filter_recordings.py --out {out} "
        f"--filter {predicate_name} {source}"
    )
    skipped_block = (
        f"Skipped {len(skipped)} episode(s) from source: "
        f"{', '.join(f'{i:03d}' for i in skipped[:20])}"
        + (f" … (+{len(skipped) - 20} more)" if len(skipped) > 20 else "")
    ) if skipped else "No episodes were skipped."

    readme = (
        f"# {out.name}\n\n"
        f"Filtered subset produced by scripts/filter_recordings.py.\n\n"
        f"## Source\n"
        f"- `{source}` — scanned {len(episodes)} episodes\n\n"
        f"## Filter\n"
        f"`{predicate_name}` — "
        + (
            "kept only episodes whose `label` column is 1.0 for every frame "
            "(no failure-labelled frames anywhere in the trajectory)."
            if predicate_name == "no-failures"
            else "kept all episodes."
        )
        + f"\n\n"
        f"Selected: **{selected}** episodes "
        + (f"(`episode_000` through `episode_{selected - 1:03d}`).\n\n" if selected else "(none).\n\n")
        + f"{skipped_block}\n\n"
        f"## Layout\n"
        f"- `data/episode_NNN.parquet` — one file per episode\n"
        f"- `videos/episode_NNN.mp4`   — one video per episode\n"
        f"- `meta/stats.json`           — scalar statistics over selected frames\n\n"
        f"## Parquet columns\n"
        f"| Column | Type | Description |\n"
        f"|--------|------|-------------|\n"
        f"| timestamp | float64 | seconds since episode start |\n"
        f"| frame_index | int64 | 0-based frame counter |\n"
        f"| observation.state_j{{1‒6}} | float | joint positions |\n"
        f"| action_j{{1‒6}} | float | joint commands |\n"
        f"| ee_x/y/z/qx/qy/qz/qw | float | EE pose |\n"
        f"| label | float | 1.0 = GOOD, 0.0 = BAD |\n\n"
        f"## Reproduce\n```\n{cmd}\n```\n"
    )
    (out / "README.md").write_text(readme)

    sample_count = stats.n.get("label", 0)
    print(
        f"\nDone. {selected} episodes selected, {len(skipped)} skipped → {out}\n"
        f"  total frames in selected: {sample_count}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the filtered dataset.")
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=repo_root / "data" / "recordings" / ".hf_cache",
        help="Cache directory for HF dataset snapshots.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty --out.")
    parser.add_argument(
        "--filter",
        default="no-failures",
        choices=list(PREDICATES.keys()),
        help="Which predicate to apply when selecting episodes.",
    )
    parser.add_argument("source", help="Local dir or HF repo id (prefix with hf: to disambiguate).")
    args = parser.parse_args(argv)

    run(args.source, args.out, args.hf_cache, args.force, args.filter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
