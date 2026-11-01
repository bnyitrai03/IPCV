# Webcam Warriors

A two-player fighting game controlled by body movement in front of one webcam.

## Requirements

- Python 3.10 – 3.13 (3.12 recommended)
- A webcam
- Internet on the first run (to install packages and download the models)

## Setup and start

Open the `IPCV_Fighting_Game` folder and:

- **Windows:** double-click `start_game.bat`
- **macOS / Linux:** run `bash start_game.command` in a terminal


Manual setup(only if the above method does not work):

```bash
cd IPCV_Fighting_Game
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python scripts/download_models.py
python main.py
```
## How to play

```bash
source .venv/bin/activate
python main.py
```

1. Stand 2–3 m from the camera, side by side, so your whole upper body and hips are visible.
2. The left person becomes **Player 1**, the right person **Player 2**. The match starts once both are recognised. After that you can move freely and swap sides.
3. Best of 3 rounds, 60 seconds each.

Moves
Uppercut --> Drive one fist quickly up and forward to head height
Block --> Hold both hands in front of your face
Special --> Raise both arms above your head and hold briefly
Move --> Lean your upper body left or right