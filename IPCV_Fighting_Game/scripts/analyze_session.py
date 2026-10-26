"""Summarise one logged session (or the newest one) into a report.

    python scripts/analyze_session.py                 # newest folder in logs/
    python scripts/analyze_session.py logs/session_20261019_203151_game

Writes ``summary.json`` and ``summary.md`` into the session folder and prints
the Markdown report. Every number comes from the CSV logs; metrics that
depend on operator annotations are labelled MANUAL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import LOGS_DIR  # noqa: E402
from evaluation.gesture_metrics import analyze_gestures  # noqa: E402
from evaluation.session import num_col, read_csv  # noqa: E402
from evaluation.tracking_metrics import analyze_tracking  # noqa: E402


def analyze_performance(d: Path) -> dict:
    rows = read_csv(d / "performance.csv")
    if not rows:
        return {}
    col = lambda k: num_col(rows, k)
    fps = col("render_fps")
    res = {"seconds_logged": len(rows),
           "render_fps_mean": float(np.nanmean(fps)),
           "render_fps_min_1s_window": float(np.nanmin(fps)),
           "low1_fps_median_of_windows": float(np.nanmedian(col("low1_fps"))),
           "vision_fps_mean": float(np.nanmean(col("vision_fps"))),
           "capture_to_result_ms_mean": float(np.nanmean(col("capture_to_result_ms"))),
           "e2e_ms_mean": float(np.nanmean(col("e2e_ms"))),
           "cpu_percent_mean": float(np.nanmean(col("cpu_percent"))),
           "rss_mb_start": float(col("rss_mb")[0]), "rss_mb_end": float(col("rss_mb")[-1])}
    # Stability over long runs: compare first and last third.
    if len(rows) >= 30:
        third = len(rows) // 3
        res["render_fps_first_third"] = float(np.nanmean(fps[:third]))
        res["render_fps_last_third"] = float(np.nanmean(fps[-third:]))
        res["rss_growth_mb"] = res["rss_mb_end"] - res["rss_mb_start"]
    return res


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"
    return str(v)


def to_markdown(name: str, report: dict) -> str:
    lines = [f"# Session report: {name}", "",
             "All values measured from the logs. `MANUAL_*` metrics rely on operator key presses "
             "(ground truth); `n/a` means the data needed was not recorded in this session.", ""]
    for section, data in report.items():
        lines.append(f"## {section}")
        lines.append("")
        def walk(prefix, obj):
            for k, v in obj.items():
                if isinstance(v, dict):
                    walk(f"{prefix}{k}.", v)
                else:
                    lines.append(f"- **{prefix}{k}**: {_fmt(v)}")
        walk("", data)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
    else:
        sessions = sorted(LOGS_DIR.glob("session_*"))
        if not sessions:
            raise SystemExit("No sessions in logs/. Run the game with --log first.")
        d = sessions[-1]
    report = {"System performance": analyze_performance(d),
              "Tracking (face / body / identity)": analyze_tracking(d),
              "Game interaction (gestures)": analyze_gestures(d)}
    (d / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    md = to_markdown(d.name, report)
    (d / "summary.md").write_text(md)
    print(md)
    print(f"\nSaved {d / 'summary.json'} and {d / 'summary.md'}")


if __name__ == "__main__":
    main()
