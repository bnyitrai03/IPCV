"""Smoke tests for the logging + analysis tooling (synthetic data, no camera)."""
import time

from config import Config
from evaluation.session import Session
from evaluation.tracking_metrics import TrackingLogger, analyze_tracking
from vision.pipeline import VisionPipeline


def test_logging_and_analysis_roundtrip(tmp_path):
    cfg = Config()
    session = Session(cfg, "test", root=tmp_path)
    logger = TrackingLogger(session)
    vp = VisionPipeline(cfg, "synthetic")
    vp.frame_listeners.append(logger.on_frame)
    vp.start()
    try:
        time.sleep(1.5)
        logger.annotate(time.perf_counter(), "f5", "stationary_start")
        time.sleep(1.0)
        logger.annotate(time.perf_counter(), "f5", "stationary_end")
        logger.annotate(time.perf_counter(), "f7", "crossing_ok")
    finally:
        vp.stop()
        session.close()
    report = analyze_tracking(session.dir)
    assert report["frames"] > 40
    assert report["player1"]["face_detection_rate_pct"] > 90
    assert report["MANUAL_crossings_ok"] == 1
    # Filtering must reduce stationary jitter compared to the raw detections.
    p1 = report["player1"]
    assert p1["face_stationary_jitter_filtered"] < p1["face_stationary_jitter_raw"]
