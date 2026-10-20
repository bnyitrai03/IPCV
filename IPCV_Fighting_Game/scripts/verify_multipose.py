"""Verify that the pose/face models really detect TWO people on this machine.

    python scripts/verify_multipose.py                 # live webcam, 20 s
    python scripts/verify_multipose.py --image two_people.jpg
    python scripts/verify_multipose.py --seconds 60 --camera 1

Shows the camera with every detected skeleton/face and prints measured
statistics (no assumed numbers): fraction of frames with 0/1/2 poses and
faces, and per-model inference time. Press Q to stop early.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config  # noqa: E402
from vision.camera import CameraThread  # noqa: E402
from vision.face_tracking import MediaPipeFaceBackend  # noqa: E402
from vision.pose_tracking import SKELETON_EDGES, MediaPipePoseBackend  # noqa: E402


def draw(img, poses, faces):
    h, w = img.shape[:2]
    cols = [(0, 200, 255), (255, 180, 0), (0, 255, 0), (255, 0, 255)]
    for i, p in enumerate(poses):
        lm = p.image
        for a, b in SKELETON_EDGES:
            if lm[a, 3] > 0.5 and lm[b, 3] > 0.5:
                cv2.line(img, (int(lm[a, 0] * w), int(lm[a, 1] * h)),
                         (int(lm[b, 0] * w), int(lm[b, 1] * h)), cols[i % 4], 3)
    for f in faces:
        pts = (f.landmarks[:, :2] * [w, h]).astype(int)
        for x, y in pts[::8]:
            cv2.circle(img, (x, y), 1, (255, 255, 255), -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int)
    ap.add_argument("--image")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.camera is not None:
        cfg.camera.index = args.camera
    pose = MediaPipePoseBackend(cfg.pose, cfg.model_path(cfg.pose.model_file))
    face = MediaPipeFaceBackend(cfg.face, cfg.model_path(cfg.face.model_file))
    pose_counts, face_counts = Counter(), Counter()
    pose_ms, face_ms = [], []

    def process(bgr, ts_ms):
        small = cv2.resize(bgr, (cfg.camera.inference_width,
                                 int(bgr.shape[0] * cfg.camera.inference_width / bgr.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        a = time.perf_counter(); poses = pose.detect(rgb, ts_ms); pose_ms.append((time.perf_counter() - a) * 1000)
        a = time.perf_counter(); faces = face.detect(rgb, ts_ms); face_ms.append((time.perf_counter() - a) * 1000)
        pose_counts[len(poses)] += 1
        face_counts[len(faces)] += 1
        disp = cv2.flip(small, 1)
        draw(disp, poses, faces)
        cv2.putText(disp, f"poses {len(poses)}  faces {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return disp

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Cannot read {args.image}")
        # VIDEO mode: feed the still image a few times so tracking settles.
        for k in range(5):
            disp = process(img, k * 40)
        if not args.no_window:
            cv2.imshow("verify_multipose", disp)
            cv2.waitKey(0)
    else:
        cam = CameraThread(cfg.camera)
        cam.start()
        t0, last = time.perf_counter(), 0
        try:
            while time.perf_counter() - t0 < args.seconds:
                fr = cam.latest(last)
                if fr is None:
                    time.sleep(0.002)
                    continue
                last = fr.frame_id
                disp = process(fr.image, int(fr.capture_ts * 1000))
                if not args.no_window:
                    cv2.imshow("verify_multipose (Q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            cam.stop()
    cv2.destroyAllWindows()
    n = sum(pose_counts.values())
    print(f"\nFrames analysed: {n}")
    for k in sorted(set(pose_counts) | {0, 1, 2}):
        print(f"  {k} poses: {pose_counts[k] / n * 100:5.1f} %    {k} faces: {face_counts[k] / n * 100:5.1f} %")
    print(f"Pose inference: mean {np.mean(pose_ms):.1f} ms, p95 {np.percentile(pose_ms, 95):.1f} ms")
    print(f"Face inference: mean {np.mean(face_ms):.1f} ms, p95 {np.percentile(face_ms, 95):.1f} ms")
    if pose_counts[2] == 0:
        print("\nWARNING: two poses were never detected. Check that both people are fully in frame.")


if __name__ == "__main__":
    main()
