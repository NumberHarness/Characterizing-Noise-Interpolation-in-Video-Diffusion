"""Local model paths for optional video generation and object tracking.

Weights are intentionally not distributed with this repository. Set the
environment variables documented in README.md or model_paths.env.example.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _configured_path(variable: str, default_relative_path: str) -> str:
    value = os.environ.get(variable)
    path = Path(value).expanduser() if value else ROOT / default_relative_path
    return str(path.resolve())


WAN22_MODEL_PATH = _configured_path(
    "WAN22_MODEL_PATH", "models/Wan2.2-TI2V-5B-Diffusers")
WAN21_MODEL_PATH = _configured_path(
    "WAN21_MODEL_PATH", "models/Wan2.1-T2V-1.3B-Diffusers")
COTRACKER_MODEL_PATH = _configured_path(
    "COTRACKER_MODEL_PATH", "models/cotracker2.pth")


def yolo_model_source() -> str:
    """Return a configured YOLO checkpoint or Ultralytics' auto-download name."""
    configured = os.environ.get("YOLO11_WEIGHTS_PATH")
    if configured:
        return require_local_model(
            configured, "YOLO11_WEIGHTS_PATH", directory=False)
    local_checkpoint = ROOT / "yolo_weights" / "yolo11x.pt"
    return str(local_checkpoint) if local_checkpoint.is_file() else "yolo11x.pt"


def require_local_model(path: str, variable: str, *, directory: bool = True) -> str:
    """Fail with setup guidance before handing a missing path to a model loader."""
    resolved = Path(path).expanduser()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(
            f"Model {kind} not found: {resolved}\n"
            f"Set {variable} to the local model {kind}; see README.md.")
    return str(resolved.resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect local model paths.")
    parser.add_argument("--check", choices=("wan22", "wan21", "cotracker"))
    args = parser.parse_args()
    configured_models = {
        "wan22": ("WAN22_MODEL_PATH", WAN22_MODEL_PATH, True),
        "wan21": ("WAN21_MODEL_PATH", WAN21_MODEL_PATH, True),
        "cotracker": ("COTRACKER_MODEL_PATH", COTRACKER_MODEL_PATH, False),
    }
    if args.check:
        variable, path, directory = configured_models[args.check]
        print(require_local_model(path, variable, directory=directory))
        raise SystemExit(0)

    for variable, path, directory in configured_models.values():
        resolved = Path(path)
        exists = resolved.is_dir() if directory else resolved.is_file()
        print(f"{variable}={path} [{ 'configured' if exists else 'not installed' }]")
    print(f"YOLO11 source={yolo_model_source()}")
