"""Canonical path resolver for the vla-safety repo.

Repo root is detected from this file's location, but can be overridden with
the VLA_SAFETY_ROOT environment variable (useful when running outside an
editable install, e.g. inside a container with a different mount path).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(
    os.environ.get("VLA_SAFETY_ROOT", Path(__file__).resolve().parents[1])
)
ASSETS: Path = REPO_ROOT / "assets"
CALIBRATION: Path = REPO_ROOT / "calibration"
CONFIGS: Path = REPO_ROOT / "configs"

__all__ = ["REPO_ROOT", "ASSETS", "CALIBRATION", "CONFIGS"]
