"""Config loader. Single entry point for all pipeline parameters."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

# Project root = parent of src/
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    """Load config.yaml and resolve all paths relative to project root."""
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve paths to absolute against ROOT for robustness.
    p = cfg["paths"]
    non_path_keys = {"raw_sheet"}  # values under 'paths' that are not filesystem paths
    for k, v in p.items():
        if k in non_path_keys:
            continue
        p[k] = str((ROOT / v).resolve())
    cfg["_root"] = str(ROOT)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create output directories if missing."""
    for key in ["processed_dir", "tables_dir", "figures_dir", "reports_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)
