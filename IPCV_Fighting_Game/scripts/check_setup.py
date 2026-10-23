"""Setup doctor: checks everything the webcam game needs, in order, and says
exactly what to fix. Run it before the game (the start scripts do this):

    python scripts/check_setup.py              # check, download missing models
    python scripts/check_setup.py --camera 1   # check a specific camera

Exit code 0 = ready to play.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"


def fail(msg: str, fix: str) -> None:
    print(f"{FAIL} {msg}")
    print(f"         FIX: {fix}")
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()
    print("\n=== Webcam Warriors setup check ===\n")

    # 1. Python
    v = sys.version_info
    if v < (3, 10):
        fail(f"Python {v.major}.{v.minor} is too old",
             "install Python 3.12 from https://www.python.org/downloads/ and recreate .venv")
    print(f"{OK} Python {v.major}.{v.minor}.{v.micro} ({platform.machine()})")

    # 2. Packages
    missing = []
    for mod in ("numpy", "cv2", "pygame", "mediapipe"):
        try:
            __import__(mod)
        except Exception as exc:          # noqa: BLE001
            missing.append(f"{mod} ({exc.__class__.__name__}: {exc})")
    if missing:
        fail("missing packages: " + ", ".join(missing),
             "activate the venv, then: pip install -r requirements.txt")
    import cv2
    import mediapipe
    print(f"{OK} packages: mediapipe {mediapipe.__version__}, opencv {cv2.__version__}")

    # 3. Models
    from config import load_config
    cfg = load_config()
    if args.camera is not None:
        cfg.camera.index = args.camera
    needed = [cfg.pose.model_file, cfg.face.model_file]
    absent = [n for n in needed if not cfg.model_path(n).is_file()]
    if absent:
        print(f"{WARN} models missing ({', '.join(absent)}) - downloading now...")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")])
        if r.returncode != 0 or any(not cfg.model_path(n).is_file() for n in needed):
            fail("model download failed",
                 "check your internet connection; on macOS also run: pip install certifi")
    print(f"{OK} models present: {', '.join(needed)}")

    # 4. Camera
    from vision.camera import CameraError, CameraThread
    cam = CameraThread(cfg.camera)
    try:
        cam.start()
    except CameraError as exc:
        others = []
        for idx in range(3):
            if idx == cfg.camera.index:
                continue
            c = cv2.VideoCapture(idx)
            if c.isOpened() and c.read()[0]:
                others.append(idx)
            c.release()
        hint = (f"camera(s) {others} DO work: start with --camera {others[0]}" if others else
                ("macOS: System Settings > Privacy & Security > Camera > enable Terminal "
                 "(or the app you run this from), then QUIT that app fully (Cmd+Q) and reopen it. "
                 "Also close FaceTime/Zoom/Teams." if platform.system() == "Darwin" else
                 "close other apps using the camera (Teams, Zoom); on Windows check Settings > Privacy > Camera"))
        fail(f"camera {cfg.camera.index} cannot be opened: {exc}", hint)
    time.sleep(0.6)
    frame = cam.latest()
    w, h = cam.actual_size
    if frame is None or frame.image.mean() < 3:
        cam.stop()
        fail(f"camera {cfg.camera.index} opened but gives black/no images",
             "uncover the lens, close other camera apps, or try --camera 1")
    print(f"{OK} camera {cfg.camera.index} works: {w}x{h}")

    # 5. Models actually run on a camera frame
    from vision.face_tracking import MediaPipeFaceBackend
    from vision.pose_tracking import MediaPipePoseBackend
    try:
        pose = MediaPipePoseBackend(cfg.pose, cfg.model_path(cfg.pose.model_file))
        face = MediaPipeFaceBackend(cfg.face, cfg.model_path(cfg.face.model_file))
    except Exception as exc:              # noqa: BLE001
        cam.stop()
        fail(f"MediaPipe could not load the models: {exc}",
             "re-download: python scripts/download_models.py --force")
    counts, times = [], []
    last = 0
    t_end = time.perf_counter() + 3.0
    print("         Stand in front of the camera for 3 seconds...")
    while time.perf_counter() < t_end:
        fr = cam.latest(last)
        if fr is None:
            time.sleep(0.005)
            continue
        last = fr.frame_id
        small = cv2.resize(fr.image, (cfg.camera.inference_width,
                                      int(fr.image.shape[0] * cfg.camera.inference_width / fr.image.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        a = time.perf_counter()
        p = pose.detect(rgb, int(fr.capture_ts * 1000))
        f = face.detect(rgb, int(fr.capture_ts * 1000))
        times.append((time.perf_counter() - a) * 1000)
        counts.append((len(p), len(f)))
    cam.stop()
    pose.close()
    face.close()
    if not counts:
        fail("no frames processed", "try again; if it repeats, send the output to your team")
    max_p = max(c[0] for c in counts)
    max_f = max(c[1] for c in counts)
    ms = sum(times) / len(times)
    print(f"{OK} models run: {ms:.0f} ms per frame (pose+face), ~{1000 / max(ms, 1):.0f} fps possible")
    if max_p == 0:
        print(f"{WARN} nobody detected - stand 2-3 m away with upper body visible, more light")
    else:
        print(f"{OK} detected up to {max_p} person(s) and {max_f} face(s)")
    if ms > 60:
        print(f"{WARN} slow; for more speed create config.json with "
              '{"pose": {"model_file": "pose_landmarker_lite.task"}} '
              "and run: python scripts/download_models.py --pose lite")
    print("\n=== READY: start the game ===\n")


if __name__ == "__main__":
    main()
