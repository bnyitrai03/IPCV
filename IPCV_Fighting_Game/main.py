"""Webcam Warriors — IPCV 2026 webcam-controlled two-player fighting game.

Usage examples::

    python main.py                         # start at the menu
    python main.py --camera 1 --log        # second camera, write CSV logs
    python main.py --vision synthetic      # no webcam: scripted synthetic players
    python main.py --start keyboard        # skip the menu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from config import load_config


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Webcam-controlled two-player fighting game")
    ap.add_argument("--config", help="JSON file with config overrides")
    ap.add_argument("--camera", type=int, help="camera index (default 0)")
    ap.add_argument("--width", type=int, help="camera capture width")
    ap.add_argument("--height", type=int, help="camera capture height")
    ap.add_argument("--vision", choices=("camera", "synthetic"), default="camera",
                    help="'synthetic' replaces the webcam by a scripted test scenario")
    ap.add_argument("--log", action="store_true", help="write evaluation CSV logs to logs/")
    ap.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--start", choices=("webcam", "keyboard", "cpu", "gesture_test"),
                    help="skip the menu and start this mode")
    ap.add_argument("--frames", type=int, help="exit after N frames (automated smoke tests)")
    ap.add_argument("--screenshot", help="save a screenshot when exiting via --frames")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.fullscreen:
        cfg.game.fullscreen = True
    if args.mute:
        cfg.game.sound = False
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from application import App   # imported after env setup

    app = App(cfg, vision_source=args.vision, log_session=args.log, debug=args.debug,
              max_frames=args.frames, auto_start=args.start, screenshot=args.screenshot)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
