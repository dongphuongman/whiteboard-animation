#!/usr/bin/env python3
"""
Streaming-stroke animation - environment bootstrap script.

Duties:
  1. Create an isolated Python venv under the skill dir (reuse if present)
  2. Check that required third-party libs import
  3. Auto-install missing libs
  4. Print ENV_PY=<interpreter path> on the last line for callers to capture

Usage:
  python prepare_env.py          # build env + fill deps, print ENV_PY
  python prepare_env.py --check  # probe only, nonzero exit if anything missing
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

# skill root = two levels up from this script
SKILL_ROOT = Path(__file__).resolve().parent.parent
VENV_ROOT = SKILL_ROOT / ".venv"

# import name -> pip package name
DEPS: dict[str, str] = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "av": "av",  # PyAV: pip-only H.264 encoding, no system ffmpeg needed
    "PIL": "Pillow",  # render_annotation_preview.py draws numbered region previews (with CJK labels)
    "edge_tts": "edge-tts",  # tts_narration.py --provider edge (free narration)
    "vieneu": "vieneu",  # tts_narration.py --provider vietneu (offline Vietnamese narration)
}


def interpreter_path() -> Path:
    """Python executable inside the venv (cross-platform)."""
    if sys.platform.startswith("win"):
        return VENV_ROOT / "Scripts" / "python.exe"
    return VENV_ROOT / "bin" / "python"


def ensure_venv(check_only: bool) -> Path:
    py = interpreter_path()
    if VENV_ROOT.exists() and py.exists():
        print(f"[ok] Reusing existing venv: {VENV_ROOT}")
        return py

    if check_only:
        print(f"[err] Venv not built yet: {VENV_ROOT}")
        sys.exit(1)

    print(f"[..] Creating venv: {VENV_ROOT}")
    venv.create(str(VENV_ROOT), with_pip=True)
    print("[ok] Venv ready")
    return py


def can_import(py: Path, import_name: str) -> bool:
    probe = subprocess.run(
        [str(py), "-c", f"import {import_name}"],
        capture_output=True,
    )
    return probe.returncode == 0


def install(py: Path, packages: list[str]) -> bool:
    if not packages:
        return True
    print(f"[..] Installing deps: {', '.join(packages)}")
    res = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", *packages],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[err] Install failed:\n{res.stderr}")
        return False
    print("[ok] Deps installed")
    return True


def main() -> None:
    check_only = "--check" in sys.argv

    py = ensure_venv(check_only)

    missing: list[str] = []
    for import_name, pip_name in DEPS.items():
        if can_import(py, import_name):
            print(f"[ok] {pip_name}")
        else:
            print(f"[miss] {pip_name}")
            missing.append(pip_name)

    if missing:
        if check_only:
            print(f"\nMissing {len(missing)} deps: {', '.join(missing)}")
            sys.exit(1)
        if not install(py, missing):
            sys.exit(1)

    # Last line: agreed output for callers to capture
    print(f"\nENV_PY={py}")


if __name__ == "__main__":
    main()
