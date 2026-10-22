#!/bin/bash
# Webcam Warriors - one-step start for macOS (and Linux).
# Run from Terminal:   bash start_game.command        (extra options are passed on,
#                      e.g.  bash start_game.command --camera 1 --fullscreen)
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ not found. Install Python 3.12 from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close..."; exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo ">> First run: creating the Python environment (takes a few minutes)..."
  "$PY" -m venv .venv || { echo "Could not create .venv"; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing packages failed (see above)."; exit 1; }
fi

CAM_ARGS=()
for ((i=1; i<=$#; i++)); do
  if [ "${!i}" = "--camera" ]; then j=$((i+1)); CAM_ARGS=(--camera "${!j}"); fi
done

.venv/bin/python scripts/check_setup.py "${CAM_ARGS[@]}" || { echo; echo "Fix the [FAIL] line above, then run this again."; read -r -p "Press Enter to close..."; exit 1; }

echo ">> Starting the game in WEBCAM mode (Esc = pause/menu)..."
.venv/bin/python main.py --start webcam "$@"
