import logging
import multiprocessing as mp
import os
import queue
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from .ble_message import Message

logger = logging.getLogger("yolo_gateway")

BASE_DIR = os.path.dirname(__file__)
MODEL_NAME = "yolov8n-pose.pt"

# YOLO pose keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

KEYPOINT_CONFIDENCE = 0.3
# Hand height, expressed in shoulder widths so it works at any distance.
PULL_THRESHOLD = 0.4   # a hand this far below the shoulders counts as pulled
STEER_DEADBAND = 0.15  # height difference between hands needed to steer

# Upper-body skeleton pairs used by the preview
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12)]

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30
PREVIEW_WIDTH = 240
PREVIEW_HEIGHT = 180
IMGSZ = 256
TORCH_THREADS = int(os.environ.get("ROWING_TORCH_THREADS", "2"))


def find_model_path():
    """Use the model from yolo_test when present, otherwise let YOLO download it."""
    candidates = [
        os.path.join(BASE_DIR, MODEL_NAME),
        os.path.join(os.path.dirname(BASE_DIR), "yolo_test", MODEL_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return MODEL_NAME


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera %s: %dx%d @ %.0f FPS",
                camera_index,
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS))
    return cap


class YoloGateway:
    """Webcam pose backend that replaces the BLE paddle input.

    Inference runs in a separate process: the pygame loop keeps its own GIL
    and is not slowed down by the model, and the model is not starved by the
    game's rendering.
    """

    def __init__(self, message: Message, camera_index=0, model_path=None, imgsz=IMGSZ):
        self.message = message
        self.camera_index = camera_index
        self.model_path = model_path or find_model_path()
        self.imgsz = imgsz
        self.running = True

        try:
            self._ctx = mp.get_context("fork")
        except ValueError:
            self._ctx = mp.get_context()

        self._updates = self._ctx.Queue(maxsize=1)
        self._stop_event = self._ctx.Event()
        self._camera_ok = self._ctx.Value("b", 0)
        self._process = self._ctx.Process(
            target=_inference_worker,
            args=(self._updates, self._stop_event, self._camera_ok, self.camera_index,
                  self.model_path, self.imgsz),
            daemon=True,
        )
        self._process.start()

        self._reader = threading.Thread(target=self._read_updates, daemon=True)
        self._reader.start()
        self.message.CONFIGURED.set()
        logger.info("YOLO Gateway started")

    def stop(self):
        self.running = False
        self._stop_event.set()
        self._reader.join(timeout=1)
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._updates.close()
        self.message.LEFT = False
        self.message.RIGHT = False
        logger.info("YOLO Gateway stopped")

    def _read_updates(self):
        while self.running:
            self.message.camera_ok = bool(self._camera_ok.value)
            if not self.message.camera_ok:
                self.message.preview = None

            try:
                left, right, fps, inference_ms, preview_bytes = self._updates.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break

            self.message.LEFT = left
            self.message.RIGHT = right
            self.message.fps = fps
            self.message.inference_ms = inference_ms
            if preview_bytes:
                frame = np.frombuffer(preview_bytes, dtype=np.uint8)
                self.message.preview = frame.reshape(PREVIEW_HEIGHT, PREVIEW_WIDTH, 3)

    @staticmethod
    def _pick_player(result):
        """Return the index of the largest person (closest to the camera)."""
        if result.boxes is None or len(result.boxes) == 0:
            return None
        xyxy = result.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(areas.argmax())

    @staticmethod
    def _keypoints_valid(result, idx):
        keypoints = result.keypoints
        if keypoints is None or keypoints.xy is None:
            return False
        if idx >= len(keypoints.xy):
            return False
        if keypoints.conf is not None:
            conf = keypoints.conf[idx].cpu().numpy()
            return all(conf[k] >= KEYPOINT_CONFIDENCE for k in (L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST))
        return True

    @staticmethod
    def _paddles_from_pose(pts):
        """Map hand heights directly to paddle states.

        Both hands pulled down -> both paddles (straight). One hand clearly
        lower than the other -> that paddle only, so the canoe turns.
        """
        l_sh, r_sh = pts[L_SHOULDER], pts[R_SHOULDER]
        l_wr, r_wr = pts[L_WRIST], pts[R_WRIST]

        shoulder_width = max(abs(r_sh[0] - l_sh[0]), 1.0)
        shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
        left_drop = (l_wr[1] - shoulder_y) / shoulder_width
        right_drop = (r_wr[1] - shoulder_y) / shoulder_width

        if max(left_drop, right_drop) < PULL_THRESHOLD:
            return False, False
        if left_drop - right_drop > STEER_DEADBAND:
            return True, False
        if right_drop - left_drop > STEER_DEADBAND:
            return False, True
        return True, True

    @staticmethod
    def _make_preview(frame, result, idx):
        """Small BGR->RGB preview with the player's box and pose drawn on it."""
        preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        if idx is None:
            return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        sx = PREVIEW_WIDTH / frame.shape[1]
        sy = PREVIEW_HEIGHT / frame.shape[0]

        if result.boxes is not None and idx < len(result.boxes):
            x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy()
            cv2.rectangle(preview, (int(x1 * sx), int(y1 * sy)),
                          (int(x2 * sx), int(y2 * sy)), (0, 255, 255), 1)

        if result.keypoints is not None and idx < len(result.keypoints.xy):
            pts = result.keypoints.xy[idx].cpu().numpy()
            for a, b in SKELETON:
                if pts[a][0] == 0 and pts[a][1] == 0:
                    continue
                if pts[b][0] == 0 and pts[b][1] == 0:
                    continue
                cv2.line(preview,
                         (int(pts[a][0] * sx), int(pts[a][1] * sy)),
                         (int(pts[b][0] * sx), int(pts[b][1] * sy)),
                         (0, 255, 0), 1)
            for x, y in pts:
                if x == 0 and y == 0:
                    continue
                cv2.circle(preview, (int(x * sx), int(y * sy)), 2, (0, 255, 0), -1)

        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def _inference_worker(updates, stop_event, camera_ok, camera_index, model_path, imgsz):
    import torch
    import ultralytics.utils.torch_utils as torch_utils

    # Ultralytics resets torch to NUM_THREADS (cpu_count - 1) on every
    # predict, which starves the game on small CPUs. Keep a small budget.
    torch_utils.NUM_THREADS = TORCH_THREADS
    torch.set_num_threads(TORCH_THREADS)
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # camera retry is noisy
    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(model_path)

    cap = open_camera(camera_index)
    if cap is None:
        logger.warning("Camera %s not available, retrying...", camera_index)
    while cap is None and not stop_event.is_set():
        camera_ok.value = 0
        time.sleep(1.0)
        cap = open_camera(camera_index)
    if cap is None:
        return
    camera_ok.value = 1

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera disconnected, waiting for it to come back...")
                cap.release()
                camera_ok.value = 0
                cap = None
                while cap is None and not stop_event.is_set():
                    time.sleep(1.0)
                    cap = open_camera(camera_index)
                if cap is None:
                    break
                camera_ok.value = 1
                prev_time = time.perf_counter()
                continue

            frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            result = model.predict(frame, verbose=False, imgsz=imgsz)[0]
            inference_ms = (time.perf_counter() - t0) * 1000.0

            idx = YoloGateway._pick_player(result)
            preview = YoloGateway._make_preview(frame, result, idx)

            if idx is None or not YoloGateway._keypoints_valid(result, idx):
                left, right = False, False
            else:
                pts = result.keypoints.xy[idx].cpu().numpy()
                left, right = YoloGateway._paddles_from_pose(pts)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0 else fps * 0.9 + inst_fps * 0.1

            payload = (left, right, fps, inference_ms, preview.tobytes())
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            try:
                updates.put_nowait(payload)
            except queue.Full:
                pass
    except Exception:
        logger.exception("Error in YOLO inference process")
    finally:
        camera_ok.value = 0
        cap.release()
