# IPCV Canoe Game

The new POC lives in [`canoe_game/`](canoe_game/README.md), following
[`system_design.md`](system_design.md). Run it with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r canoe_game/requirements.txt
python -m canoe_game
```

Choose **KEYBOARD RUN** to play without a camera. Webcam setup, controls,
architecture, test instructions and known limitations are in
[`canoe_game/README.md`](canoe_game/README.md).

`rowing_game/` remains as the read-only reference implementation.
