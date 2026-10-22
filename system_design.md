# Canoe Game - System Design

- **Course:** IPCV-2026-1A, University of Twente
- **Document status:** v0.2, design decisions locked; implementation planning
- **Date:** 2026-10-21
- **Author role:** Task 5 owner (scene processing and real-time integration) / team lead
- **Target:** A new, from-scratch two-player cooperative canoe game that satisfies all five compulsory technical tasks in `Assignment_Description.md`.

This version incorporates the full `grilling` interview. Game-design and architecture decisions are settled. The remaining open items are external dependencies (team roster, deadline, model version, asset rights) and numeric tuning.

---

## 0. Decision log

| Topic | Decision |
|---|---|
| Codebase strategy | Clean rewrite in `canoe_game/`; copy assets; keep `rowing_game/` as read-only reference |
| Player setup | Two players share one canoe, stand side by side facing one webcam |
| Paddle sides | **Free side switching**: the grip top-hand selects the side; either player may take either side; both may share a side |
| Side grip | Right hand on top = port/left paddle; left hand on top = starboard/right paddle |
| Side lock | The side is locked for each stroke; a player may switch sides between strokes after recovery |
| Stroke model | Full catch -> drive -> recovery; only the drive emits propulsion; holding the hands low does not row |
| Turning | Physical differential: a starboard stroke turns the canoe left; a port stroke turns it right |
| Perfect sync | Only a port + starboard pair from two different players within 300 ms counts; it gives a forward bonus, score multiplier, and face-effect charge |
| Same-side pair | Two strokes on the same side still power-turn the canoe, but give no sync, score bonus, or face charge |
| Second action | **Brace**: both players raise both arms straight up and hold; both must be braced at a crocodile lunge impact to deflect it |
| Brace effects | Braced players cannot row; brace works only against crocodiles, not rocks or banks; entering brace resets the shared rhythm streak/charge |
| Canoe motion | Fixed screen y near the lower third; the world scrolls; rowing changes forward speed and lateral position |
| Run structure | Endless run; one collision with a rock, crocodile, or bank ends the run |
| Practice / grace | No practice mode, no start grace, no special restart; the standard `R` restart on the game-over overlay remains |
| Difficulty | Old obstacle scoring plus distance and perfect-sync bonuses; slow ramp reaching maximum pressure around 3 minutes |
| Identity loss | **Limp mode**: 0.5 s grace, then freeze the lost player's strokes/stats/effects; remaining player can switch sides freely; no sync pair; 30% forward-speed penalty; crocodile lungers disabled; both lost -> auto-pause |
| Identity ambiguity | Continue side-based control; freeze P1/P2 attribution, stats, and face effects; never guess or swap |
| Readiness | Webcam mode shows a short readiness screen; auto-start after 3 s of two players, two faces, and valid grips; no calibration |
| Calibration | **None**; default shoulder-width-normalised thresholds |
| Tutorial | **None**; no How-to-Play panel and no instruction text; only the live side/grip indicator |
| Face effect | Minimal sync ring + counter; shows the balanced-sync streak/charge; resets on brace; extensible effect layer |
| Face effect reward | Score multiplier + visual charge; no speed surge or damage immunity |
| CV stack | MediaPipe Pose Landmarker (up to 2 poses) + Face Landmarker/FaceMesh (up to 2 faces); custom identity tracker; no YOLO |
| Performance fallback | Pose Lite model, lower resolution, face every other pose frame; keyboard remains the emergency fallback |
| Layout | 1280x720 split: 800x600 game viewport + 480x720 camera/telemetry panel |
| Keyboard | Testing + emergency fallback; same control-event API; keys for each side and brace |
| Player coordinates | Normalised play-area coordinates plus shoulder-width depth; optional metric calibration only for the report |
| Assets | Reuse old visuals/SFX; replace `menu.mp3`; generate crocodile and paddler sprites (including both-side and brace frames) |
| Team | Five-person group; author owns Task 5; task contracts are drafted but not frozen until the roster is known |

---

## 1. Concept

Two players share one canoe. Either player can paddle on either side of the canoe. The grip top-hand tells the game which side is being paddled: right hand on top means the left/port paddle, left hand on top means the right/starboard paddle. Players may switch sides between strokes, and both may paddle on the same side.

A full rowing cycle is required for propulsion. The players coordinate their strokes to go straight (one port and one starboard stroke in sync), to turn (one side alone, or both on the same side for a power turn), and to brace against crocodiles by raising both arms straight up.

The game is a top-down, pixel-art endless river run. Rocks, crocodiles, and the river banks are lethal. One collision ends the run. The score rewards distance, obstacles passed, and balanced (opposite-side) perfect syncs. Face-anchored sync rings show the rhythm streak and make the cooperation visible.

### Design pillars

1. **Cooperation is mechanical.** Balanced sync is the fastest, safest way to travel; same-side power turns and solo paddling are useful but costly.
2. **The body is the controller.** Rowing is a repeated full-body motion, not a hand-height threshold held forever.
3. **Free but readable.** Sides are free, so the camera panel must make the current side and grip validity obvious at a glance.
4. **Robust real-time behaviour.** Default thresholds, confidence gating, and limp mode keep the game playable when tracking degrades.
5. **Extensible CV/effects.** Grip/side detection, identity, and face effects are modules with clear interfaces.

### Non-goals

- Realistic hydrodynamics or a full 3D canoe simulation.
- Additional cameras, IMUs, gamepads, or other hardware.
- Face recognition or biometric identification.
- A level/finish-line campaign, practice mode, or health system.
- A How-to-Play tutorial; the live indicator is the only in-game teaching aid.

---

## 2. Assignment compliance matrix

| Compulsory task | Game feature | Primary modules | Evidence to demonstrate |
|---|---|---|---|
| 1. Face tracking and augmented effects | Two faces tracked; smoothed sync ring + counter anchored to each face; balanced-sync charge; reset on brace | `cv/face.py`, `cv/identity.py`, `ui/effects.py` | Two players move, turn, approach/leave the camera; rings stay aligned and never switch |
| 2. Body pose estimation and motion tracking | Two skeletons; shoulder/elbow/wrist keypoints; free-side grip detection; stroke and brace state machines; confidence gating; filtering | `cv/pose.py`, `input/stroke_detector.py`, `input/brace_detector.py` | Live skeleton, side indicator, stroke counter, brace state; missing-keypoint and low-confidence demo |
| 3. Multiplayer detection and identity tracking | Two simultaneous players; persistent P1/P2; face-pose association; crossing, occlusion, limp mode, re-entry; normalised spatial output | `cv/identity.py`, `cv/readiness.py` | Two players cross; labels, stats, and face effects stay with the correct person; limp mode |
| 4. Player interaction and game control | Free side selection; row and brace actions; balanced sync; power turns; croc deflection; feedback | `input/`, `game/control.py`, `game/scoring.py`, `ui/hud.py` | Both players independently row on either side; balanced sync goes straight; same-side power turn; croc brace succeeds/fails |
| 5. Scene processing and real-time integration | Scrolling curved river, rocks, crocodiles, wake/splash effects, camera panel, HUD, performance instrumentation | `app/`, `game/`, `ui/`, `cv/worker.py` | Complete game runs continuously at target FPS; measured latency; controlled failure handling |

The camera panel is the shared window into Tasks 1-3; the game viewport is the shared window into Tasks 4-5.

---

## 3. Reuse from the old game

The old game is an 800x600 pygame application. It is single-player and tracks only the largest detected person. The following logic is intentionally reused or adapted.

### 3.1 River generation and turning

From `rowing_game/river.py`:

- The river is a polyline of 5 px vertical segments.
- Each new segment advances `curve_phase` by `0.02`.
- The centreline offset is `sin(curve_phase) * 100` plus a uniform random value in `[-5, 5]`.
- The offset is clamped to `[-120, 120]`.
- The river is 400 px wide on an 800 px screen.
- Scrolling pops segments from the front and appends new ones at the back.
- `get_river_bounds_at_y(y)` returns the left and right bank bounds for collision and spawning.

The new game keeps this generator so the river still bends like the old one. Constants become configuration values because the canoe and crocodile sprites are larger.

### 3.2 Rock spawning and obstacle scrolling

From `rowing_game/game_core.py` and `rowing_game/obstacle.py`:

- Obstacles spawn at `y = -OBSTACLE_HEIGHT`.
- The spawn x is random inside the river bounds at the top, with a 20 px bank margin.
- The base spawn interval is 2.0 s; easy is 3.0 s, hard is 1.0 s.
- Obstacles move only with the river scroll.
- Passing an obstacle awards +1 point once.
- Obstacles are removed after they leave the bottom of the screen.

The new game reuses this spawn placement and scoring model, adds crocodiles, and drives the scroll from the current plus the canoe's forward speed.

### 3.3 Collision and game over

From `rowing_game/canoe.py` and `rowing_game/game_core.py`:

- The canoe collision hitbox is 70% of the sprite size.
- Rock collision uses AABB.
- Bank collision currently samples the river bounds at the canoe centre only; the new game samples the canoe's front and rear corners as well.
- Hitting a rock, hitting a bank, or being pushed off the bottom ends the old game.

The new game ends the run on the first collision with a rock, crocodile, or bank. The old "pushed off the bottom" condition is removed because the canoe's vertical screen position is fixed.

### 3.4 Assets

| Asset | Native size | Reuse |
|---|---|---|
| `boat.png` | 482x906 RGBA | Reuse as the shared canoe, scaled with nearest-neighbour; draw two paddlers on top |
| `rock.png` | 779x818 RGBA | Reuse as the rock obstacle |
| `water.png` | 1024x1024 | Reuse as the tiled water texture |
| `grass.png` | 1024x1024 | Reuse as the tiled bank texture |
| `menu.png` | 1536x1024 | Reuse as the menu background |
| `game_start.mp3`, `game_over.mp3`, `point.mp3`, `river_splashy.mp3` | MP3 | Reuse as SFX/background, subject to rights confirmation |
| `menu.mp3` | MP3 | Replace; metadata indicates a third-party meme remix and no licence file exists |

New assets required: top-down crocodile sprite with drift/lunge frames, two paddler sprites or one two-colour sprite sheet with port/starboard paddle frames and a both-arms-up brace frame, splash/wake particles, and an optional croc ripple.

---

## 4. Game rules

### 4.1 Run structure

1. Player selects **Webcam** or **Keyboard (testing)** in the menu.
2. In webcam mode, a short readiness screen appears. It waits for two players, two faces, and valid grips. After 3 s of continuous readiness, the run auto-starts.
3. A short countdown starts the run.
4. The canoe is carried down an endless river. Rocks, crocodiles, and banks must be avoided.
5. The first collision ends the run.
6. The results overlay shows score, distance, best score, and per-player stats. `R` restarts; `Q` returns to the menu. There is no start grace and no special instant restart.

Keyboard mode skips the readiness screen and starts directly. It is clearly labelled as testing/fallback.

### 4.2 Actions

Each player has two distinguishable actions:

| Action | Physical execution | Detection | Game effect |
|---|---|---|---|
| Row | Full catch -> drive -> recovery with both hands on the imaginary paddle; top hand selects the side | Wrist trajectory + top-hand grip + confidence FSM | Forward impulse plus side impulse |
| Brace | Both arms raised straight up above the head and held | Both wrists above the head/shoulder line, arms extended, held for 0.4 s | Prevents propulsion; contributes to the crocodile shield |

The grip convention:

- **Port (left paddle):** right hand on top, left hand below.
- **Starboard (right paddle):** left hand on top, right hand below.
- Either player may use either grip. Both players may use the same side.
- The side is locked for a stroke. A player may switch sides between strokes after recovery.

### 4.3 Canoe control

Let `P_port` and `P_starboard` be the summed power of completed strokes on each side during the current update, in `[0, 1]`.

- **Forward impulse:** `speed += (P_port + P_starboard) * K_FORWARD`.
- **Lateral impulse:** `vx += (P_port - P_starboard) * K_LATERAL`.
- **Direction:** a port stroke moves the canoe right; a starboard stroke moves it left. This matches the interview answer: rowing only on the right makes the boat turn left.
- **Balanced pair:** a port and a starboard stroke from two different players within `SYNC_WINDOW` cancel the side impulses, receive a sync forward multiplier, and emit a perfect-sync event.
- **Same-side pair:** two strokes on the same side add their lateral impulses, produce a strong power turn, and receive no sync bonus, no score bonus, and no face charge.
- **Single stroke:** normal forward impulse plus a lateral impulse; steering without a sync bonus.
- **No strokes:** the canoe keeps drifting with the current. The river never stops, so doing nothing is not safe.
- **Drag:** lateral velocity is damped every frame; forward speed decays toward the current speed.
- **Visual tilt:** the canoe sprite tilts/rotates slightly according to lateral velocity; this is cosmetic and does not change the collision model.

The canoe's screen y is fixed near the lower third of the game viewport. The river and obstacles scroll downward at `BASE_CURRENT + canoe_forward_speed`. This gives the players reaction time and makes lateral steering the core challenge.

### 4.4 Perfect sync and rhythm

- A perfect-sync pair consists of one port stroke and one starboard stroke from **two different players** within `SYNC_WINDOW` (initial 300 ms).
- A single player's strokes never count as a sync, even if they use both sides within the window.
- Two strokes on the same side never count as a sync, even if they are simultaneous.
- A perfect sync grants:
  - a forward impulse multiplier,
  - a score multiplier/streak bonus,
  - a pulse on both face sync rings.
- The shared rhythm streak/charge increases on each perfect sync. It resets after `SYNC_TIMEOUT` without a perfect sync, or immediately when either player enters brace.
- The face ring shows the shared rhythm streak. Per-player perfect-sync participation is shown in the camera-panel stats.

### 4.5 Brace and crocodiles

Two crocodile behaviours are used:

1. **Drifter:** swims with the current and oscillates laterally. It is a moving obstacle; steering is the answer.
2. **Lunger:** telegraphs a lunge by turning toward the canoe and creating a ripple for `CROC_TELEGRAPH` (initial 0.8 s), then lunges.

Brace rules:

- A player is braced when both wrists are above the head/shoulder line, the arms are straight, and the pose is held for `BRACE_HOLD` (initial 0.4 s).
- A braced player cannot emit strokes.
- The canoe is shielded only when both players are braced at the moment the lunge reaches the canoe.
- A successful brace deflects the crocodile, awards a bonus, and produces a splash/shield effect.
- A failed brace or no brace ends the run.
- Brace works only against crocodiles. Rocks, banks, and drifters are still lethal.
- Entering brace resets the shared rhythm streak and face ring charge.
- While one player is lost (limp mode), crocodile lungers are disabled, so a lone player is never presented with an unavoidable both-brace test.

### 4.6 Obstacles and spawning

- Rocks reuse the old spawn logic: random x inside the top river bounds, 20 px margin, one hazard per spawn interval.
- Crocodiles use the same placement rules, with a larger sprite and a reduced collision hitbox.
- The spawner must guarantee a passable gap: never place a rock and a crocodile in the same narrow band that leaves no route; enforce a minimum vertical gap between hazards.
- Crocodile lunges must have enough telegraph time and spawn distance to be braceable.
- Obstacles are removed after leaving the bottom of the viewport. Obstacles above the viewport after a restart are cleared.

### 4.7 Scoring

- `+1` per obstacle passed (rock or crocodile), reusing the old rule.
- `+1` per fixed distance interval (for example, every 10 m).
- Perfect-sync bonus: a score multiplier/streak bonus applied to obstacle and distance points; same-side power turns receive no sync bonus.
- `+CROC_BLOCK_BONUS` per successfully braced crocodile lunge.
- Per-player stats: strokes, port/starboard distribution, perfect-sync participation, braces, tracking-loss count, near misses.
- The shared score is the headline. Per-player stats are displayed on the results screen and in the camera panel.
- The best score is persisted.

### 4.8 Difficulty

- Reuse the old `easy / normal / hard` spawn intervals: 3.0 s / 2.0 s / 1.0 s.
- Add a slow endless ramp: the spawn interval decreases and the crocodile probability increases with distance.
- The ramp reaches maximum pressure around 3 minutes and then caps.
- The obstacle mix and gaps must remain fair at maximum pressure.

### 4.9 Loss, limp mode, and identity safety

- If a player's pose/face confidence drops, hold the last valid state for `PLAYER_LOST_GRACE` (initial 0.5 s).
- After the grace period, mark the player as **LOST** and enter limp mode:
  - that player's strokes, per-player stats, and face effect freeze;
  - the remaining player can row either side and switch grips freely;
  - no perfect-sync pair can be formed;
  - forward speed receives a 30% penalty;
  - crocodile lungers are disabled;
  - a clear "P1 lost" warning appears.
- If both players are lost, auto-pause. Resume when both are re-acquired and a short countdown finishes.
- If a lost player returns, re-acquire their identity by appearance/position. If re-acquisition is ambiguous, keep attribution frozen until resolved.
- If identities are ambiguous while both players are visible, continue side-based control (the side is known from the grip), but freeze per-player attribution, stats, and face effects and show "identifying". Never guess or swap.
- The shared score continues during limp mode and ambiguity; per-player attribution resumes only when identity is certain.
- Never fabricate a stroke or a brace from stale or missing keypoints.

---

## 5. Controls and signal processing

### 5.1 Free-side grip detection

The side is determined from the grip, not from fixed player ownership.

- **Port:** right wrist above left wrist; both wrists close together; valid two-handed grip.
- **Starboard:** left wrist above right wrist; both wrists close together; valid two-handed grip.
- A stroke's side is locked at its start. A player can change sides only after completing recovery and establishing the new grip.
- A short stability window (`GRIP_STABLE`, initial 150 ms) prevents accidental side flips.
- If the grip is invalid or ambiguous, no stroke is emitted and the camera panel shows an invalid-grip icon.
- Both players may use the same side. The game sums side-specific impulses.

### 5.2 Pose normalisation

All body measurements are expressed in a body-relative frame:

- Origin: shoulder midpoint.
- Scale: shoulder width.
- Rotation: align the shoulder line with the horizontal axis using the torso angle from the hip midpoint.

This makes the default thresholds invariant to camera distance, player height, and small leans. There is no per-player calibration; the defaults are tuned across multiple testers and body sizes.

MediaPipe landmark indices used:

- Nose: 0
- Shoulders: 11, 12
- Elbows: 13, 14
- Wrists: 15, 16
- Hips: 23, 24

### 5.3 Stroke state machine

The detector is a deterministic state machine driven by smoothed, confidence-gated keypoints. It is testable with recorded keypoint fixtures.

States:

- `SEARCH`: looking for a valid grip and a raised/forward catch pose.
- `ARMED`: grip valid, hands in the catch region, side locked.
- `DRIVE`: hands are moving down/back above the speed threshold.
- `RECOVER`: hands are returning to the catch region after a stroke.

Rules:

- A stroke is emitted only on the `DRIVE -> RECOVER` transition after sufficient travel.
- Power is derived from normalised travel distance and drive duration, clamped to `[0, 1]`.
- Hysteresis thresholds prevent chatter at the catch and finish boundaries.
- A cooldown (`STROKE_COOLDOWN`, initial 350 ms) prevents double-counting.
- If confidence is lost for longer than `KEYPOINT_HOLD` (initial 200 ms), the state resets without emitting a stroke.
- A recovery must complete before the next stroke; repeatedly flicking the hands does not generate events.
- Holding the hands low stays in `RECOVER` or `SEARCH`; it does not emit continuous strokes.
- The emitted `StrokeEvent` carries `player_id`, `side`, `power`, `source`, and `timestamp_ms`.

### 5.4 Brace state machine

- Enter `BRACING` when both wrists are above the head/shoulder line, the arms are extended, and the pose is held for `BRACE_HOLD` (initial 400 ms).
- Stay braced while the arms remain up.
- Exit when the arms drop below the head/shoulder line or the pose becomes unreliable.
- A braced player cannot emit strokes.
- Entering brace resets the shared rhythm streak and face ring charge.
- The game considers the canoe shielded only when both players are braced at the lunge impact.

### 5.5 Synchronisation tracker

- Each valid stroke is timestamped with the source frame time.
- A perfect-sync pair requires one port and one starboard stroke from two different players within `SYNC_WINDOW` (initial 300 ms).
- Same-side pairs are not sync pairs; they are power turns.
- A single player's two strokes never form a sync pair.
- Perfect syncs increment the shared rhythm streak and the face-effect charge.
- The streak resets after `SYNC_TIMEOUT` (initial 1.5 s) without a perfect sync, or immediately when either player enters brace.
- The control layer applies a forward multiplier to perfectly paired strokes; unpaired strokes still generate individual thrust and turning.

### 5.6 Keyboard fallback

Keyboard input uses the same `StrokeEvent` and `BraceState` interfaces as the webcam pipeline, so the game logic is identical. Because sides are free, each player has a key for each side plus a brace key.

| Key | Action |
|---|---|
| `A` | P1 port stroke (one event per key press) |
| `D` | P1 starboard stroke |
| `S` | P1 brace (hold) |
| `Left` | P2 port stroke |
| `Right` | P2 starboard stroke |
| `Down` | P2 brace (hold) |
| `R` | Restart after game over |
| `Esc` | Pause / menu |
| `F1` | Debug overlay (includes the key legend) |

Keyboard mode skips camera capture, readiness, and model loading entirely. It is selected in the menu and clearly labelled "Keyboard (testing / fallback)."

### 5.7 Debug tools

- Live skeleton with landmark confidence values.
- Per-player state labels: `SEARCH / ARMED / DRIVE / RECOVER / BRACING / LOST`.
- Current side, grip validity, stroke power, sync offset, shared rhythm streak.
- CV FPS, pose latency, face latency, end-to-end event latency.
- Optional synthetic keypoint playback for deterministic tests.

---

## 6. Computer-vision pipeline

### 6.1 Camera and worker

- Capture: 640x480, MJPG, 30 FPS, buffer size 1, mirrored horizontally for display consistency.
- The CV pipeline runs in a separate process (or a dedicated worker with a latest-result queue) so pygame rendering and game logic never block on inference.
- The worker publishes an immutable `CvFrame` containing the downscaled RGB preview, player tracks, face tracks, grip hints, diagnostics, and timestamps.
- The game process consumes only the latest frame; stale frames are dropped. This bounds latency under load.

### 6.2 Pose: MediaPipe Pose Landmarker

- Use the MediaPipe Tasks API `PoseLandmarker` in `VIDEO` running mode.
- Run a single model instance with `num_poses=2`; this provides the two-player data path without loading two models.
- Model bundle: `pose_landmarker_lite.task` for CPU responsiveness (fallback to `full` if accuracy is insufficient).
- Key landmarks: shoulders, elbows, wrists, hips, nose, eyes, ears.
- Confidence gates: `min_pose_detection_confidence`, `min_pose_presence_confidence`, and `min_tracking_confidence` are configuration values; landmark visibility/presence is checked per keypoint before use.
- The model has no persistent player IDs. Identity is assigned by the custom tracker in Section 6.4.

If the chosen MediaPipe build cannot return two poses in one call, the fallback is a two-stage pipeline: a lightweight person detector proposes two ROIs, and one `PoseLandmarker` instance runs per ROI. This is slower and should be avoided on the CPU-only demo laptop.

### 6.3 Face: MediaPipe Face Landmarker / FaceMesh

- Use the MediaPipe Tasks API `FaceLandmarker` in `VIDEO` running mode with `num_faces=2`.
- Run on the full frame when both faces are expected; run every other pose frame if latency demands it.
- Outputs: 478 face landmarks, optional blendshapes, optional facial transformation matrices.
- Head orientation: use the facial transformation matrix when available; otherwise solve PnP against a canonical face model.
- Face landmarks are mapped back to full-frame coordinates for drawing.
- Face results are associated with pose tracks by head-region proximity and size (Section 6.4).

### 6.4 Identity tracking

MediaPipe does not provide persistent IDs, so the new game implements a small identity manager.

For each pose detection:

1. Compute a body bounding box, shoulder midpoint, and torso colour histogram.
2. Predict each existing track's position with a constant-velocity model.
3. Match detections to tracks with a Hungarian assignment using:
   - IoU of body boxes,
   - normalised centre distance,
   - torso colour-histogram similarity,
   - a soft initial-seat prior (P1 was initially left, P2 initially right).
4. Update the matched track; create a tentative track for unmatched detections; age unmatched tracks.
5. Promote a tentative track to active after a few consistent frames.
6. Mark a track lost after the grace period; keep its identity memory for re-entry.

For each face detection:

1. Compute the face bounding box and nose/eye centres.
2. Compare with each pose track's head region (nose, eyes, ears).
3. Run a 2x2 Hungarian assignment by centre distance and size difference.
4. Bind the face to the pose identity; a face that cannot be matched is ignored.

The initial identity assignment happens on the readiness screen: the leftmost player becomes P1, the rightmost P2. This is only an identity label; it does not fix paddle sides. If the players cross, identities follow the people, while each stroke's side comes from the current grip.

If both faces and poses are ambiguous, the game continues side-based control but freezes attribution, stats, and face effects until the identities resolve. It never guesses.

### 6.5 Calibrated spatial output (Task 3)

Each active player track outputs:

- `image_xy`: normalised image coordinates.
- `world_xy`: play-area coordinates in a normalised or metric frame.
- `depth_m`: estimated distance from shoulder width and a pinhole camera model.
- `identity`: `P1` / `P2` / `UNKNOWN`.
- `confidence`: tracking confidence.

There is no in-game metric calibration. The default output is normalised play-area coordinates plus shoulder-width depth. An optional one-time floor-plane calibration can be performed offline for the report (four tape marks), but it is not part of the run flow.

Task 4 uses the identity and the stroke-side grip to route actions and attribute stats. The spatial output is displayed in the camera panel and logged for Task 3's evaluation; it supports crossing/re-entry decisions and proves that identities are stable.

### 6.6 Filtering

- **Face pose and landmarks:** One Euro filter; separate cutoff frequencies for position and rotation.
- **Pose keypoints:** One Euro or a short moving average, tuned so the stroke detector remains responsive.
- **Confidence:** a keypoint is valid only above its visibility/presence threshold. Invalid keypoints are held for a short window, then cause a controlled state reset.
- **Prediction:** constant-velocity prediction is used only for identity matching, never to fabricate game actions.

### 6.7 MediaPipe resource plan

- Model files are stored under `canoe_game/models/` and downloaded by a documented setup script or first-run helper.
- Pin the MediaPipe version in `requirements.txt`; verify Python 3.12 compatibility on the demo laptop.
- Record model file names, versions, and checksums in the README and AI/external-tools statement.
- If model loading fails, the app shows a clear error in the camera panel and offers keyboard mode; it must not crash.

---

## 7. Architecture

### 7.1 Package layout

```
canoe_game/
  __init__.py
  __main__.py
  app/
    main.py              # process startup, app loop, state dispatch
    states.py            # Menu, Readiness, Countdown, Playing, Paused, Results
    settings.py          # validated persistent settings
    config.py            # tunable constants
  cv/
    worker.py            # CV process, latest-result queue, lifecycle
    camera.py            # capture, mirroring, timestamps
    pose.py              # MediaPipe Pose Landmarker wrapper
    face.py              # MediaPipe Face Landmarker wrapper
    identity.py          # custom pose/face identity tracker
    readiness.py         # two-player/face/grip readiness status
    filters.py           # One Euro, confidence hold, utilities
    types.py             # CvFrame, PoseTrack, FaceTrack, diagnostics
  input/
    events.py            # StrokeEvent, GripState, BraceState, ControlFrame
    base.py              # InputSource protocol
    cv_source.py         # CV frame -> control events
    stroke_detector.py   # per-player rowing FSM, free-side grip
    brace_detector.py    # per-player brace FSM
    sync.py              # opposite-side pairing and rhythm streak
    keyboard.py          # keyboard input source
  game/
    session.py           # run lifecycle, game-over, restart, limp mode
    control.py           # canoe physics from control events
    canoe.py             # sprite, collision rect, tilt, wake
    river.py             # curved river generation, scrolling, bounds
    obstacles.py         # rocks, crocodiles, spawner, fairness checks
    scoring.py           # score, distance, per-player stats
    difficulty.py        # spawn ramp and hazard mix
  ui/
    menu.py
    readiness_screen.py
    hud.py
    camera_panel.py
    effects.py           # extensible face/game effects
    results.py
    theme.py
    debug_overlay.py
  assets/
    images/
    sounds/
  models/
    README.md            # acquisition/version instructions
  tests/
    fixtures/
    test_grip.py
    test_stroke_detector.py
    test_brace_detector.py
    test_sync.py
    test_control.py
    test_river.py
    test_spawner.py
    test_scoring.py
    test_identity.py
    test_limp_mode.py
  tools/
    evaluate_cv.py       # recorded-clip evaluation and metrics
    record_session.py    # optional diagnostics capture
  requirements.txt
  README.md
  OWNERS.md              # pending roster
```

### 7.2 Data flow

```
Webcam
  -> CV process: capture -> pose -> face -> identity -> CvFrame
  -> latest-result queue
  -> game process: CvFrame -> grip/side -> stroke/brace detectors
  -> ControlEvents -> CanoeController -> GameSession -> renderer
Keyboard
  -> KeyboardInputSource -> same ControlEvents
```

The game process never calls MediaPipe directly. The CV process never draws the game. The only shared data is the latest `CvFrame` (and diagnostics), which keeps the interfaces narrow and testable.

### 7.3 Application state machine

```
BOOT -> MENU -> READINESS -> COUNTDOWN -> PLAYING -> PAUSED
                                      \-> RESULTS -> MENU / PLAYING
```

- `MENU`: start, input mode, difficulty, settings, quit.
- `READINESS` (webcam only): live camera; waits for two players, two faces, and valid grips; auto-starts after 3 s of continuous readiness. No calibration, no tutorial text.
- `COUNTDOWN`: 3-2-1; all inputs ignored until the run begins.
- `PLAYING`: game loop, CV events, HUD.
- `PAUSED`: resume, switch to keyboard, return to menu.
- `RESULTS`: score, best score, per-player stats, restart, menu.

### 7.4 Core interfaces

```python
@dataclass(frozen=True)
class Keypoint:
    x: float          # normalised image coordinate
    y: float
    z: float          # relative depth from MediaPipe, if available
    visibility: float
    presence: float

@dataclass(frozen=True)
class PoseTrack:
    player_id: str            # "P1" | "P2" | "UNKNOWN"
    bbox: tuple[float, float, float, float]
    keypoints: dict[int, Keypoint]
    world_xy: tuple[float, float] | None
    depth_m: float | None
    confidence: float
    timestamp_ms: int
    lost: bool

@dataclass(frozen=True)
class FaceTrack:
    player_id: str
    bbox: tuple[float, float, float, float]
    landmarks: tuple[Keypoint, ...]
    head_pose: tuple[float, float, float]  # yaw, pitch, roll in degrees
    confidence: float
    timestamp_ms: int
    lost: bool

@dataclass(frozen=True)
class CvFrame:
    preview_rgb: bytes
    preview_size: tuple[int, int]
    poses: tuple[PoseTrack, ...]
    faces: tuple[FaceTrack, ...]
    camera_ok: bool
    pose_fps: float
    face_fps: float
    pose_ms: float
    face_ms: float
    total_ms: float
    timestamp_ms: int

@dataclass(frozen=True)
class GripState:
    player_id: str
    side: str          # "port" | "starboard"
    valid: bool
    stable_ms: int
    timestamp_ms: int

@dataclass(frozen=True)
class StrokeEvent:
    player_id: str
    side: str          # "port" | "starboard"
    power: float       # 0..1
    source: str        # "webcam" | "keyboard"
    timestamp_ms: int

@dataclass(frozen=True)
class BraceState:
    player_id: str
    active: bool
    timestamp_ms: int
```

The `StrokeDetector` and `BraceDetector` are pure state machines over keypoint frames, so they can be unit-tested without a camera. The `CanoeController` is a pure function of control events and delta time, so it can be tested deterministically.

### 7.5 Threading and process model

- Main process: pygame, app state, game logic, rendering, audio.
- CV process: camera capture, MediaPipe pose, MediaPipe face, identity tracking, preview overlay data.
- Communication: a `multiprocessing.Queue(maxsize=1)` or a shared latest-result object; old results are discarded.
- A supervisor thread in the main process monitors the CV process and restarts it after a crash with backoff. If restart fails repeatedly, the UI offers keyboard mode.
- Keyboard mode does not start the CV process.

### 7.6 Settings and persistence

Persist in `canoe_game/settings.json`:

- input mode (`webcam` / `keyboard`)
- difficulty (`easy` / `normal` / `hard`)
- music and SFX volume
- player display names (optional)
- high score
- debug overlay toggle
- last camera index

Load defensively: clamp volumes to 0-100, validate difficulty against a whitelist, validate player names, and fall back to defaults on malformed JSON. The old settings loader trusted arbitrary JSON; the new one must not.

### 7.7 Logging and metrics

- Python `logging` to console and `canoe_game/logs/app.log`.
- Per-frame CV timings and game FPS are sampled, not logged every frame.
- Optional CSV session log: timestamp, FPS, pose latency, face latency, identity switches, grip/side switches, stroke counts, sync offsets, brace events, collisions.
- The debug overlay shows the same counters live.

---

## 8. Scene, rendering, and UI

### 8.1 Window layout

1280x720 window:

```
+--------------------------------------------------+---------------------------+
|                                                  |  CAMERA PANEL (480x720)   |
|             GAME VIEWPORT (800x600)              |  +---------------------+  |
|             centred vertically at y=60           |  | live preview 480x360|  |
|                                                  |  +---------------------+  |
|  river, canoe, rocks, crocs, HUD                  |  P1 card: side, grip,     |
|                                                  |  strokes, sync, status    |
|                                                  |  P2 card: side, grip,     |
|                                                  |  strokes, sync, status    |
|                                                  |  CV FPS / latency         |
+--------------------------------------------------+---------------------------+
```

- The game viewport renders at the old 800x600 resolution and is not stretched. This preserves the old pixel-art scale and the old river/obstacle constants.
- The camera panel is 480x720. The preview is 480x360 (4:3), with 360 px below for player cards, face-ring charge, rhythm streak, and diagnostics.
- The preview shows the face rings, pose skeleton, stroke/brace indicators, identity colours/labels, and current side per player.
- There is no instruction text. The live side/grip indicator is the only in-game teaching aid.

### 8.2 River and banks

- Reuse the old procedural sine river.
- Draw water with the tiled `water.png` texture and banks with `grass.png`.
- Add a subtle wake and flow lines driven by the canoe speed.
- Sample bank collision at the canoe's front and rear corners, not just the centre.
- The river curve is the primary steering challenge even without obstacles.

### 8.3 Canoe and paddlers

- Scale `boat.png` with nearest-neighbour to fit two paddlers (initial target: about 90x160 px).
- Draw two paddler sprites on the benches: P1 in blue, P2 in orange.
- Paddler art includes port/starboard stroke frames and a both-arms-up brace frame, matching the free-side mechanic.
- Tilt/rotate the canoe slightly with lateral velocity.
- On a stroke, spawn a splash on the corresponding side and a small wake particle.
- On a collision, play a splash/debris effect and shake the game viewport.

### 8.4 Rocks and crocodiles

- Rocks reuse `rock.png`, scaled to the obstacle size, with a reduced collision hitbox.
- Crocodiles are top-down sprites with a tail and drift/lunge frames.
- Croc drifters oscillate laterally while moving downstream.
- Croc lungers show a telegraph (snout turn, ripple, sound) before accelerating toward the canoe.
- Spawn fairness: maintain a passable gap, avoid overlapping hazards, and limit simultaneous hazards.
- Lungers are disabled while one player is lost (limp mode).
- Obstacles are removed after leaving the bottom of the viewport; obstacles above the viewport after a restart are cleared.

### 8.5 Effects

Game-scene effects:

- Water splash on each stroke.
- Wake trail behind the canoe.
- Croc ripple and lunge telegraph.
- Screen shake and debris on collision.
- "PERFECT SYNC" text burst and a brief river-spirit glow around the canoe on a balanced sync.
- "POWER TURN" cue for a same-side pair (no sync charge).
- Shield/splash effect on a successful croc brace.
- Optional speed lines at high forward speed.

Face effects:

- **Minimal sync ring + counter**: a colour-coded ring anchored to the face landmarks, plus a numeric counter for the shared rhythm streak/perfect-sync charge.
- Aligned through face landmarks and head pose; smoothed; fades out on face loss.
- Pulses on a perfect sync. Resets on brace. Shows a shield state while braced.
- Bound to the pose identity; never switches between players.
- The effect layer is intentionally extensible.

```python
class FaceEffect(Protocol):
    def update(self, state: EffectState, dt: float) -> None: ...
    def draw(self, surface, anchor: FaceAnchor) -> None: ...

class EffectState:
    player_id: str
    rhythm_streak: int
    rhythm_charge: float
    bracing: bool
    lost: bool
    ambiguous_identity: bool
```

New effects can be added without changing the CV pipeline or the game rules. The v1 implementation is the minimal ring/counter; headdress or mask variants are future work.

### 8.6 HUD and camera panel

Game HUD:

- Score, distance, forward speed.
- Shared rhythm meter, perfect-sync streak, score multiplier.
- Current side per player, with a side-switch animation.
- Brace/shield indicator (both braced).
- "LOST" warning and limp-mode indicator.
- "PERFECT SYNC" and "POWER TURN" feedback.
- Game-over overlay with score, best, distance, per-player stats, and `R`/`Q`.

Camera panel:

- Live preview with skeleton, face ring/counter, identity labels, stroke/brace icons.
- P1/P2 cards: current side, grip-valid icon, strokes, port/starboard distribution, perfect-sync participation, tracking status, face-ring charge.
- CV FPS and latency.
- No instruction text; the side/grip icons and labels are status indicators.

### 8.7 Menus and results

- Main menu: Start, Settings, Quit. Background: `menu.png`. Input mode selection (Webcam / Keyboard testing) and difficulty are in Settings or on Start.
- No How-to-Play panel and no control-instruction text. The live side/grip indicator in the readiness screen and camera panel is the only teaching aid.
- Readiness screen (webcam): live camera, P1/P2 labels, face/grip status icons, 3-second auto-start countdown. No calibration controls, no instructions.
- Settings: input mode, difficulty, music/SFX volume, debug overlay.
- Results: shared score, best, distance, per-player stats, replay/menu.
- Pause: resume, switch to keyboard, return to menu.

---

## 9. Performance and evaluation

### 9.1 Targets

| Metric | Target |
|---|---|
| Game render/update | 60 FPS |
| Pose inference | at least 15 FPS on the demo laptop |
| Face inference | at least 10 FPS on the demo laptop |
| End-to-end input latency | under 150 ms from physical stroke to game response |
| CV process crash recovery | automatic restart within 2 s |
| Soak test | 10 minutes without crash or unbounded memory growth |

The demo machine is CPU-only (AMD Ryzen 5 7520U, 8 threads, no NVIDIA GPU). The old project's own commit states 10 FPS. The new design treats responsiveness as a first-class constraint.

### 9.2 Instrumentation

- Timestamps at capture, pose start/end, face start/end, identity end, event emission, and game application.
- Rolling averages and worst-case values.
- Live debug overlay.
- Optional CSV/JSON session log for the report.
- Bottleneck breakdown by stage.

### 9.3 Evaluation experiments

1. **Pose latency and stability:** static and dynamic rowing; jitter before/after filtering; FPS.
2. **Free-side grip detection:** side classification accuracy for both grips; switch latency; invalid-grip rejection; both players on the same side.
3. **Stroke detection:** precision/recall against labelled clips; missed strokes; false strokes from recovery; missing-keypoint behaviour.
4. **Brace detection:** true/false positives; response time; both-arms-up pose; behaviour when one arm is occluded.
5. **Sync:** opposite-side pair detection, same-side pair rejection, single-player pair rejection, sync-window sensitivity, brace reset.
6. **Face tracking:** landmark jitter, head-pose stability, alignment under translation/rotation/distance changes.
7. **Identity:** two players crossing, partial occlusion, temporary disappearance, re-entry; count identity switches; limp mode; ambiguity attribution freeze.
8. **End-to-end:** game FPS, CV FPS, input-to-action latency, frame drops during CV load.
9. **No-calibration robustness:** multiple testers of different heights, distances, and clothing; default-threshold performance.
10. **Failure cases:** camera unplugged, model missing, low light, one player leaves, both players leave, keyboard fallback.

### 9.4 Acceptance criteria for the demo

- Two players are detected simultaneously and keep P1/P2 labels.
- Both can row on either side; the live side indicator updates on a grip switch.
- Opposite-side sync goes straight and triggers the sync ring/pulse; same-side sync power-turns without a sync bonus.
- Both players can brace; a telegraphed lunge is deflected when both are braced.
- A player crossing or briefly leaving does not swap controls or face effects; limp mode is visible and recovers.
- The game runs continuously for at least 3 minutes in the exhibition configuration.
- The camera panel and game viewport are both visible in the 60-second video.
- The measured FPS/latency overlay is shown.

---

## 10. Testing

### 10.1 Unit tests

- Grip/side detector: both grips, invalid grips, side stability, side switching.
- Stroke FSM: catch, drive, finish, recovery, cooldown, hysteresis, missing keypoints, low confidence.
- Brace FSM: both arms up, hold, release, interruption.
- Sync tracker: opposite-side pairing, same-side rejection, same-player rejection, timeout, brace reset.
- Canoe controller: forward/lateral impulses, power turn, balance, drag, clamping, turn direction.
- River: segment generation, bounds, scrolling, bank collision with corner sampling.
- Spawner: bounds, margins, fairness gaps, difficulty intervals, off-screen cleanup.
- Scoring: obstacle pass, distance, balanced-sync bonus, no same-side bonus, croc block, per-player stats.
- Identity: matching, crossing, occlusion, loss, re-entry, ambiguous attribution freeze (synthetic tracks).
- Limp mode: freeze, speed penalty, no lungers, both-lost pause, recovery.
- Settings: validation and defaults.

Use pytest. Headless pygame tests set `SDL_VIDEODRIVER=dummy`. CV model tests should be mocked or use small recorded fixtures so the suite does not require a camera.

### 10.2 Integration tests

- Keyboard end-to-end: start -> run -> collision -> results -> restart.
- CV replay: recorded clip/landmark fixture -> `CvFrame` sequence -> grip/side -> strokes -> game state.
- Readiness flow: two synthetic players, identity assignment, grip validation, 3 s auto-start.
- Free-side flow: P1 switches sides between strokes; both players share a side; power turn.
- Limp-mode flow: freeze, speed penalty, no lungers, recovery.
- Ambiguity flow: control continues, attribution/effects freeze, resolution.
- Performance smoke test: 10-minute run with CV mocked or live.

### 10.3 Manual demo matrix

- Two players, normal lighting, both visible.
- Players cross positions.
- One player steps out and returns (limp mode).
- One player turns their head away; face ring fades and recovers.
- Both players switch sides mid-run.
- Both players row on the same side (power turn) and on opposite sides (balanced sync).
- Low light / backlight.
- Camera unplugged; keyboard fallback.
- Croc lunge with both braced and with only one braced.
- Rock/bank collision and game-over overlay.

### 10.4 Assignment demo checklist

- Show webcam input and game output simultaneously.
- Show skeletons, face rings, identity labels, and stroke/brace recognition.
- Show both players controlling the canoe, switching sides, and a balanced perfect sync.
- Show a difficult tracking condition (crossing, occlusion, or limp mode/re-entry).
- Show the measured FPS/latency overlay.
- Keep the recording one continuous take.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No calibration + free side switching | Default thresholds may misclassify sides for some body types/distances | Shoulder-width normalisation, grip-stability window, live side indicator, tune on multiple testers, keyboard fallback |
| CPU cannot run pose + face at target FPS | Laggy controls, failed demo | Pose Lite, downscale input, face every other frame, process isolation, latest-result queue, keyboard fallback |
| MediaPipe lacks multi-pose in the pinned version | Cannot track two players in one call | Verify before implementation; fallback to a person detector + two ROI pose runs |
| Identity swap when players cross | Stats/effects switch | Custom tracker with appearance, face association, ambiguity freeze; never guess |
| Limp mode gives one player too much/too little control | Unfair or trivial gameplay | 30% speed penalty, no sync, no lungers, clear HUD; tune |
| No tutorial | Players do not discover free-side switching | Prominent live side/grip indicator, side-switch animation, colour-coded paddlers; accept a steeper learning curve |
| No practice mode or start grace | Short demos | Skill-based design, keyboard fallback, careful onboarding during the live demo; no rule change |
| Brace resets rhythm | Harsh defence trade-off | Clear ring reset feedback; tune croc lunge frequency |
| Only opposite-side sync counts | Same-side pairs feel unrewarded | Visible "POWER TURN" feedback; strong turn is its own reward; no score/charge by design |
| Face ring jitter or misalignment | Task 1 looks weak | One Euro filtering, landmark-based anchor, head-pose alignment, short-loss fade |
| Unfair obstacle spawns | Frustrating demo | Gap/fairness checks, one hazard per spawn, minimum reaction distance |
| Croc lunge unavoidable | Instant game over | Long telegraph, brace window, spawn distance, disabled during limp mode |
| Old assets have no licence file | Submission risk | Confirm rights; replace suspicious audio; document asset sources |
| MediaPipe/model download fails at exhibition | No webcam mode | Bundle model files in the ZIP (small), provide setup script and checksums, keyboard fallback |
| Camera unavailable | No game | Clear error, keyboard fallback, automatic retry |
| Team interfaces diverge | Integration delays | Draft contracts now; freeze and assign in `OWNERS.md` when the roster is known |

---

## 12. Milestones

| Milestone | Scope | Exit criterion |
|---|---|---|
| M0 - Skeleton | Package layout, app states, settings, keyboard input, readiness screen, old river/rock prototype | Launch menu, complete readiness, play a rock-dodging run with keyboard |
| M1 - Game core | Free-side canoe control, grip/side events, power turns, balanced sync, scoring, game over, HUD, croc prototype | Keyboard two-player run with side switching is playable end to end |
| M2 - Pose | Camera worker, MediaPipe pose, identity v1, grip/side detector, stroke detector, readiness auto-start | Live skeleton and free-side strokes drive the canoe |
| M3 - Face | MediaPipe face, association, smoothing, sync ring v1, camera panel | Two aligned rings, perfect-sync pulse, no switching |
| M4 - Integration | Croc lunge + brace, brace reset, rhythm/score, limp mode, ambiguity freeze, effects, results, keyboard fallback | Full webcam run with all five tasks visible |
| M5 - Evaluation | Metrics harness, tests, performance tuning, failure handling | Targets met; report/poster/video material ready |

The author owns Task 5, so integration is not deferred to M4. Every milestone ends with an integrated, runnable build.

---

## 13. Open dependencies (not design decisions)

1. **Team roster and deadline.** The assignment requires five named task owners. Add names/tasks to `OWNERS.md` when available; freeze the draft contracts then.
2. **MediaPipe version and model bundle.** Pin a tested version; verify Python 3.12 compatibility; bundle `pose_landmarker_lite.task` and `face_landmarker.task` with checksums.
3. **Asset rights.** Confirm rights to the old visuals and SFX. Replace `menu.mp3`. Document all sources in the AI/external-tools statement.
4. **Numeric tuning.** Sync window, stroke thresholds, brace hold, spawn intervals, croc mix, score multipliers, speed penalty, and ramp curve are initial values to tune during M2-M5.
5. **Report, poster, and video.** Task 5 owns integration evidence: FPS/latency, failure cases, and the end-to-end pipeline description.

---

## Appendix A - Initial tuning constants

These are starting values, not final gameplay tuning.

| Constant | Initial value |
|---|---|
| Window | 1280x720 |
| Game viewport | 800x600 at (0, 60) |
| Camera panel | 480x720 at (800, 0) |
| Game FPS | 60 |
| Camera capture | 640x480, 30 FPS, MJPG |
| Pose target | >= 15 FPS, 2 poses, Lite model |
| Face target | >= 10 FPS, 2 faces, every 2nd pose frame |
| Sync window | 300 ms |
| Sync timeout | 1.5 s |
| Stroke cooldown | 350 ms |
| Grip stability | 150 ms |
| Keypoint hold | 200 ms |
| Brace hold | 400 ms |
| Player lost grace | 500 ms |
| Limp-mode speed penalty | 30% |
| Readiness auto-start | 3 s continuous |
| River segment height | 5 px |
| River curve phase step | 0.02 |
| River sine amplitude | 100 px |
| River max centre offset | 120 px |
| River width | 400 px (tunable for the larger canoe) |
| Base spawn interval | 2.0 s |
| Easy / hard spawn interval | 3.0 s / 1.0 s |
| Difficulty ramp cap | ~3 minutes |
| Rock size | 80x80 px |
| Croc size | ~80x120 px (sprite dependent) |
| Croc telegraph | 0.8 s |
| Canoe size | ~90x160 px (sprite dependent) |
| Canoe collision | 70% of sprite AABB |
| Base current speed | 1.5-2.0 px/frame equivalent |
| Max forward speed | 5-6 px/frame equivalent |
| Max lateral speed | 300-400 px/s |
| P1 colour | Blue |
| P2 colour | Orange |

---

## Appendix B - Interface contracts for task owners

Task 5 owns the contracts below. They are draft until the roster is known; freeze them then.

| Owner | Contract | Inputs | Outputs |
|---|---|---|---|
| Task 1 | `FaceTracker` / `FaceEffect` | RGB frame, pose head ROIs | `FaceTrack` list, aligned face-ring anchors |
| Task 2 | `PoseTracker`, `StrokeDetector`, `BraceDetector` | RGB frame, `CvFrame` | `PoseTrack` list, `GripState`, `StrokeEvent`, `BraceState` |
| Task 3 | `IdentityManager`, `Readiness` | Pose/face detections | stable `player_id`, `world_xy`, `depth`, lost/ambiguous/reacquired status |
| Task 4 | `ControlMapper`, `CanoeController`, `Scoring` | `GripState`, `StrokeEvent`, `BraceState`, sync events | canoe state, score, feedback events |
| Task 5 | `AppLoop`, `CvWorker`, `GameSession`, `Renderer` | all of the above | integrated real-time application, metrics, failure handling |

Integration rules:

- No task owner imports another task's private implementation; use the dataclasses and protocols.
- The game loop must never block on CV; use the latest `CvFrame`.
- Every rejected/missing detection must produce a defined state, never an exception or an invented action.
- All tunable thresholds live in `app/config.py`, not scattered through modules.
- Every task must provide a measurable evaluation script or log and a documented failure mode.

---

## Appendix C - MediaPipe setup notes

- Use the MediaPipe Tasks Python API, not the legacy single-person `mp.solutions.pose` API, for multi-pose support.
- Models:
  - `pose_landmarker_lite.task` (primary CPU model)
  - `pose_landmarker_full.task` (accuracy comparison)
  - `face_landmarker.task`
- Store models locally under `canoe_game/models/`; do not rely on a network download during the exhibition.
- Document the exact MediaPipe version, model URLs, and checksums in the README and AI tool statement.
- Verify the API on the demo laptop with a two-person test clip before integrating.
- If multi-pose output is unavailable, implement the documented person-detector + per-player ROI fallback rather than silently dropping a player.

---

## Appendix D - Assignment ownership and deliverables

| Compulsory task | Owner | Status |
|---|---|---|
| 1. Face tracking and augmented effects | TBD | Draft contract: `FaceTracker` / `FaceEffect` |
| 2. Body pose estimation and motion tracking | TBD | Draft contract: `PoseTracker` / `StrokeDetector` / `BraceDetector` |
| 3. Multiplayer detection and identity tracking | TBD | Draft contract: `IdentityManager` / `Readiness` |
| 4. Player interaction and game control | TBD | Draft contract: `ControlMapper` / `CanoeController` / `Scoring` |
| 5. Scene processing and real-time integration | Author | Integration lead; owns the app loop, CV worker, renderer, and performance evidence |

The final README and report must name all five owners consistently. `OWNERS.md` is the single source of truth once the roster is confirmed.
