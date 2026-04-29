"""
Merge multiple flat-format recording datasets into a single combined dataset.

Each input dataset must follow the layout produced by scripts/recorder.py:

    <source>/
      data/episode_NNN.parquet
      videos/episode_NNN.mp4
      meta/stats.json
      README.md

Sources can be local directories or Hugging Face dataset repo IDs (resolved
via `snapshot_download` into a local cache before merging).

Usage:
    python scripts/merge_recordings.py \\
        --out data/recordings/exp_merged \\
        data/recordings/exp_01 \\
        data/recordings/exp_02 \\
        hf:various-and-sundry/domino-trajectories-on-new-table

A source is treated as a Hugging Face dataset id when it starts with `hf:` or
when the literal string is not an existing local path and looks like
`<owner>/<name>`.

Output:
    <out>/data/episode_000.parquet ... episode_NNN.parquet  (renumbered)
    <out>/videos/episode_000.mp4   ... episode_NNN.mp4      (renumbered)
    <out>/meta/stats.json          (Welford-merged from sources)
    <out>/README.md                (auto-generated)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path


HF_PREFIX = "hf:"
EPISODE_RE = re.compile(r"^episode_(\d+)\.parquet$")
ALLOW_PATTERNS = ["data/*.parquet", "videos/*.mp4", "meta/*.json", "README.md"]


# ── Source resolution ──────────────────────────────────────────────────────


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


def validate_source(path: Path) -> None:
    if not (path / "data").is_dir():
        raise FileNotFoundError(f"{path} has no data/ subdirectory")
    if not (path / "videos").is_dir():
        raise FileNotFoundError(f"{path} has no videos/ subdirectory")
    stats = path / "meta" / "stats.json"
    if not stats.is_file():
        raise FileNotFoundError(f"{path} has no meta/stats.json")


def list_episodes(src: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted((src / "data").glob("episode_*.parquet")):
        m = EPISODE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


# ── Stats merging (Welford parallel) ──────────────────────────────────────


def _combine_stats(a: dict, b: dict) -> dict:
    n_a = a.get("count", 0) or 0
    n_b = b.get("count", 0) or 0
    if n_a == 0:
        return dict(b)
    if n_b == 0:
        return dict(a)

    n = n_a + n_b
    mean_a = a["mean"]
    mean_b = b["mean"]
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n

    M2_a = (a.get("std", 0.0) or 0.0) ** 2 * n_a
    M2_b = (b.get("std", 0.0) or 0.0) ** 2 * n_b
    M2 = M2_a + M2_b + delta * delta * n_a * n_b / n
    std = math.sqrt(M2 / n) if n > 0 else 0.0

    return {
        "min": min(a["min"], b["min"]) if a.get("min") is not None and b.get("min") is not None else (a.get("min") if a.get("min") is not None else b.get("min")),
        "max": max(a["max"], b["max"]) if a.get("max") is not None and b.get("max") is not None else (a.get("max") if a.get("max") is not None else b.get("max")),
        "mean": mean,
        "std": std,
        "count": n,
    }


def merge_stats(stats_paths: list[Path]) -> dict:
    merged: dict[str, dict] = {}
    for p in stats_paths:
        src = json.loads(p.read_text())
        for col, s in src.items():
            if col in merged:
                merged[col] = _combine_stats(merged[col], s)
            else:
                merged[col] = dict(s)
    return merged


# ── Main merge ─────────────────────────────────────────────────────────────


def merge(sources: list[str], out: Path, hf_cache: Path, force: bool) -> None:
    out_data = out / "data"
    out_videos = out / "videos"
    out_meta = out / "meta"

    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(
                f"Output directory {out} already exists and is non-empty. "
                f"Re-run with --force to overwrite."
            )
        shutil.rmtree(out)

    for d in (out_data, out_videos, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Resolve all sources to local paths.
    print("Resolving sources:", flush=True)
    resolved: list[tuple[str, Path]] = []
    for s in sources:
        path = resolve_source(s, hf_cache)
        validate_source(path)
        resolved.append((s, path))
        print(f"  • {s} → {path}", flush=True)

    # 2. Plan + copy episodes.
    next_idx = 0
    per_source_counts: list[tuple[str, int]] = []
    for label, src in resolved:
        episodes = list_episodes(src)
        videos_dir = src / "videos"
        for _orig_idx, parquet_path in episodes:
            dst_name = f"episode_{next_idx:03d}"
            shutil.copy2(parquet_path, out_data / f"{dst_name}.parquet")
            mp4 = videos_dir / parquet_path.name.replace(".parquet", ".mp4")
            if mp4.is_file():
                shutil.copy2(mp4, out_videos / f"{dst_name}.mp4")
            else:
                print(f"  ⚠ no video for {label}/{parquet_path.name}", flush=True)
            next_idx += 1
        per_source_counts.append((label, len(episodes)))
        print(f"  ✓ {label}: copied {len(episodes)} episodes", flush=True)

    total = next_idx

    # 3. Merge stats.
    merged_stats = merge_stats([src / "meta" / "stats.json" for _, src in resolved])
    (out_meta / "stats.json").write_text(json.dumps(merged_stats, indent=2))

    # 4. README.
    sources_block = "\n".join(
        f"- `{label}` — {n} episodes" for label, n in per_source_counts
    )
    cmd = "python scripts/merge_recordings.py --out " + str(out) + " " + " ".join(sources)
    readme = (
        f"# {out.name}\n\n"
        f"Merged dataset produced by scripts/merge_recordings.py.\n\n"
        f"## Sources\n{sources_block}\n\n"
        f"Total episodes: **{total}** "
        f"(`episode_000` through `episode_{total - 1:03d}`).\n\n"
        f"## Layout\n"
        f"- `data/episode_NNN.parquet` — one file per episode\n"
        f"- `videos/episode_NNN.mp4`   — one video per episode\n"
        f"- `meta/stats.json`           — Welford-merged scalar statistics\n\n"
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

    # 5. Summary.
    counts = {col: s.get("count", 0) for col, s in merged_stats.items()}
    sample_count = next(iter(counts.values()), 0)
    print(
        f"\nDone. {total} episodes → {out}\n"
        f"  stats sample count: {sample_count}\n"
        f"  per-source: {per_source_counts}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the merged dataset.")
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=repo_root / "data" / "recordings" / ".hf_cache",
        help="Cache directory for HF dataset snapshots.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty --out.")
    parser.add_argument("sources", nargs="+", help="Local dirs or HF repo ids (prefix with hf: to disambiguate).")
    args = parser.parse_args(argv)

    merge(args.sources, args.out, args.hf_cache, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
