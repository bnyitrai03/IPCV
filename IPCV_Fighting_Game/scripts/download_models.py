"""Download the MediaPipe model files into assets/models/.

    python scripts/download_models.py            # pose (full) + face
    python scripts/download_models.py --pose lite  # faster, less accurate pose model

Models are published by Google under the Apache 2.0 licence
(https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
"""
from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "assets" / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"
URLS = {
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def _context() -> ssl.SSLContext:
    try:
        import certifi   # ships with pip on most installs; fixes python.org macOS builds
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(name: str, force: bool = False) -> None:
    dest = MODELS / name
    if dest.is_file() and dest.stat().st_size > 1000 and not force:
        print(f"  ok   {name} (already present, {dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  get  {name} ...", end="", flush=True)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(URLS[name], context=_context(), timeout=60) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 16):
                fh.write(chunk)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f" FAILED: {exc}")
        print("    Tip (macOS, python.org Python): run 'Install Certificates.command' from the\n"
              "    Python folder in Applications, or 'pip install certifi', then retry.\n"
              f"    Or download manually: {URLS[name]}\n    and save it as {dest}")
        raise SystemExit(1)
    tmp.replace(dest)
    print(f" done ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=("lite", "full", "heavy"), default="full")
    ap.add_argument("--all", action="store_true", help="download every pose variant")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    names = ["face_landmarker.task"]
    names += ([f"pose_landmarker_{v}.task" for v in ("lite", "full", "heavy")] if args.all
              else [f"pose_landmarker_{args.pose}.task"])
    print(f"Downloading to {MODELS}")
    for n in names:
        download(n, args.force)
    if args.pose != "full" and not args.all:
        print(f"\nNote: config.py uses pose_landmarker_full.task by default. To use the "
              f"{args.pose} model create config.json with:\n"
              f'  {{"pose": {{"model_file": "pose_landmarker_{args.pose}.task"}}}}')


if __name__ == "__main__":
    sys.exit(main())
