```
 ██████╗ ██████╗  ██████╗ ██╗  ██╗██╗     ██████╗ ███████╗
██╔════╝██╔═══██╗██╔═══██╗██║ ██╔╝██║    ██╔═══██╗██╔════╝
██║     ██║   ██║██║   ██║█████╔╝ ██║    ██║   ██║███████╗
██║     ██║   ██║██║   ██║██╔═██╗ ██║    ██║   ██║╚════██║
╚██████╗╚██████╔╝╚██████╔╝██║  ██╗██║    ╚██████╔╝███████║
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═════╝ ╚══════╝
```

Automated cooking robot OS — computer vision ingredient monitoring, Arduino motor control, YAML-driven recipes, and a live web control dashboard.

> *COOKi grasps what every kitchen philosopher already knows: garlic turning golden is not a timer event. It is a photometric phenomenon. And anyone who tells you otherwise deserves overcooked pasta.*

## A Sneak Peek into the Journey
I have designed two iterations of COOKi, each incorporating structural refinements and 3D-printed hardware integrated with a commercial hotpot base. The first prototype featured an N20 motor with a single-arm configuration, which showed promise but had durability limitations. The second generation upgraded to a dual-arm harness design with an enhanced high-torque motor, delivering superior performance and reliability.
**version_1**
<video controls src="WhatsApp Video 2026-04-14 at 4.21.58 PM.mp4" title="COOKi_rev1"></video>
<video controls src="WhatsApp Video 2026-04-14 at 4.21.58 PM (1).mp4" title="COOKi_rev1"></video>

**version_2**


## Philosophy

**The Core Idea**

Every recipe is a directed graph. Every node is not "an ingredient" — it is a cooking action with that ingredient: a pre-condition (ingredient present), a motor behavior (how to stir it), a vision-based exit condition (what "done" looks like for this ingredient), and a fallback timer that activates when vision uncertainty gets cowardly. The orchestrator walks the graph and hands control to each node's stir-monitor loop until the node declares itself complete.

**Two Engineering Improvements**

1. **Temporal Voting** — Don't act on a single frame. The vision model runs per-frame, but you need N consecutive frames to agree before triggering the exit. This prevents a brief flash of golden light from fooling the garlic node.
2. **Confidence Hysteresis** — Each frame-vote only counts if it clears a tunable confidence threshold. Together, these make your exit conditions robust.

**Node Anatomy**

Every node has six components:
- a prompt (system asks operator to add the ingredient and waits for confirmation)
- a motor profile (the stirring strategy)
- a per-node vision classifier (binary: TARGET_STATE vs NOT_YET)
- a frame voter (requires N consecutive agreements)
- a confidence threshold
- a fallback timer

The execution loop: **stir for `stir_s` seconds** → **pause motor** → **grab frame** → **run inference** → **if voter says done, exit** → **else stir again**.

**Recipe as YAML** — The recipe is declared, not hardcoded. This makes it portable, editable, and trainable without touching code.

---

## What's in the box

| File | What it does |
|---|---|
| `main.py` | Entry point — launches UI server then runs the recipe |
| `orchestrator.py` | Recipe execution engine — walks nodes, manages motor + vision |
| `state.py` | Thread-safe live state shared between orchestrator and UI |
| `vision.py` | Unified vision loader — regression and YOLO backends, same interface |
| `trainer.py` | **Regression trainer** — live camera capture + slider labeling |
| `yolo_integration.py` | **YOLO trainer** — train from image folders (simpler, recommended) |
| `motor.py` | Arduino serial bridge — sends `speed,direction` commands |
| `ui_server.py` | Flask server — camera MJPEG, SSE state stream, motor API |
| `templates/dashboard.html` | Full web control dashboard |
| `cooki_motor.ino` | Arduino firmware — flashes to Uno/Nano |
| `recipes/pasta.yaml` | Example recipe — one-pot pasta |

---

## Hardware

| Component | Notes |
|---|---|
| Raspberry Pi 4 (2 GB+) or Jetson Nano | Main compute |
| USB webcam | Mounted directly above the pan |
| Arduino Uno or Nano | Motor PWM driver |
| DC motor + motor driver board | L298N, DRV8833, TB6612FNG, or L9110S |
| Motor arm + spatula | Your custom hardware harness |

### Arduino wiring

```
Arduino D9  →  Motor driver PWM / IN1 input
Arduino D8  →  Motor driver DIR / IN2 input
Arduino GND →  Motor driver GND
External 6-12V →  Motor driver VM (motor power rail)
```

Flash `cooki_motor.ino` before first use. The Arduino listens at 9600 baud
and replies `OK spd=200 dir=1` on each command for debugging.

---

## Installation (Raspberry Pi)

```bash
# 1. Clone or copy the cooki/ folder to your Pi

# 2. Create a virtualenv (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Note: the project uses `opencv-python-headless` so it runs cleanly in
# containers/headless Linux environments. `trainer.py` falls back to a
# terminal-driven capture loop when no display server is available.

# 4. Find your Arduino serial port
ls /dev/tty*
# Usually /dev/ttyUSB0 (USB-serial adapter) or /dev/ttyACM0 (native USB)
```

---

## Training a vision model

Every VISION node in a recipe needs a trained `.pt` model file.
COOKi has two training methods — use whichever fits your workflow.

---

### Method A — YOLO from image folders  *(recommended, simpler)*

Collect photos with your phone or download from Google Images.
Sort them into percentage-labeled subfolders. Train in one command.

**Folder structure:**

```
training_images/garlic_golden/
  0_raw/        ← photos of completely raw garlic  (0% cooked)
  25_light/     ← just starting to soften, no colour change yet
  50_half/      ← turning translucent, faint golden edges
  75_mostly/    ← clearly golden, not yet at peak
  100_done/     ← fully golden-brown (stop here — any darker = burnt)
```

**Rules:**
- Each subfolder name **must start with the percentage number**
- Any of: `.jpg  .jpeg  .png  .bmp  .webp`
- Aim for **20–50 images per class**, from different angles / lighting / fill levels
- Mix your actual pan with reference photos for better generalisation

**Train:**

```bash
python yolo_integration.py \
    --node   garlic_golden \
    --images training_images/garlic_golden

# With CUDA (much faster):
python yolo_integration.py \
    --node   garlic_golden \
    --images training_images/garlic_golden \
    --epochs 80 \
    --device cuda
```

Saves model to: `models/garlic_golden_yolo.pt`

In the recipe YAML:
```yaml
model: models/garlic_golden_yolo.pt
model_type: yolo
```

**How YOLO inference works:**
YOLOv8n-cls outputs a probability per class. COOKi computes a weighted average:

```
Σ (class_percentage × class_probability)

e.g.  50% → 0_raw (prob 0.1) + 25% → 25_light (prob 0.1) + 75% → 75_mostly (prob 0.8)
      = 0×0.1 + 0.25×0.1 + 0.75×0.8  = 0.625  →  62.5% cooked
```

This gives smooth float output — identical interface to the regression model.

---

### Method B — Regression from live camera  *(more nuanced)*

Capture frames directly from the Pi camera while you actually cook the dish.
Label each frame by sliding a 0–100% slider. Trains a MobileNetV3-Small model.

Use this when:
- Your specific pan, lighting, and ingredient brand matter a lot
- You want the model trained on exact real conditions
- You can dedicate a cooking session to data collection

**Capture session:**

```bash
python trainer.py --node garlic_golden
```

OpenCV window opens showing the camera feed.

| Key | Action |
|---|---|
| `0`–`9` | Set label (key `3` = 30% cooked, key `8` = 80%) |
| `[` / `]` | Nudge label ±5% |
| `SPACE` | Capture current frame at current label |
| `T` | Stop capture and train now |
| `Q` | Quit without training |

**Capture strategy:**
1. Start with garlic raw, label 0%, capture 3-4 frames
2. Add garlic to hot oil, capture every 15-20 seconds as it cooks
3. Label honestly based on visual appearance
4. At the golden-brown moment, capture 6-8 frames at 80–95%
5. Press `T` — training runs immediately

Aim for **60–120 frames** spread across the full 0–100% spectrum.

**Retrain without recapturing:**

```bash
python trainer.py --node garlic_golden --train-only --epochs 40
```

Saves model to: `models/garlic_golden.pt`

In the recipe YAML:
```yaml
model: models/garlic_golden.pt
model_type: regression
```

---

### Which nodes to train for the pasta recipe

| Node ID | Recommended method | What to capture |
|---|---|---|
| `garlic_golden` | YOLO | Raw garlic → golden-brown stages |
| `tomato_paste_reduced` | Either | Raw paste → darkened, reduced paste |
| `pasta_done` | YOLO | Dry pasta → al-dente → soft stages |

---

## Running a recipe

```bash
# Full run with Arduino:
python main.py recipes/pasta.yaml

# Specify port (Pi often uses ttyACM0):
python main.py recipes/pasta.yaml --port /dev/ttyACM0

# Mock mode — no Arduino, motor commands printed:
python main.py recipes/pasta.yaml --mock

# Headless — confirm only via web UI (no keyboard needed):
python main.py recipes/pasta.yaml --headless

# Mock camera — synthetic frames (no webcam required):
python main.py recipes/pasta.yaml --mock-cam

# Both (perfect for testing without hardware):
python main.py recipes/pasta.yaml --mock --headless

# Full no-hardware dev mode (no Arduino + no webcam):
python main.py recipes/pasta.yaml --mock --mock-cam --headless

# Custom camera or port:
python main.py recipes/pasta.yaml --cam 1 --ui-port 8080
```

Open the dashboard at **http://[pi-ip]:5000** from any device on the same network.

---

## Web dashboard

The dashboard is a single-page app that connects to the Pi over your local network.
Open it on your phone while standing at the stove.

### What's on screen

**Camera feed** — live MJPEG stream with % cooked badge burned in during
cooking. A subtle amber scanning line animates while a VISION node is active.

**Prompt banner** — appears when a node is waiting. Shows the operator instruction
(e.g. "Add garlic to the oil") and a large **CONFIRM ✓** button.

**Live stats** — four cards updating every 300ms:
- **% Cooked** — current vision model output
- **Votes** — consecutive agreeing frames vs target (e.g. 3/5)
- **Node time** — elapsed time in current node
- **Session** — total cook time

**Confidence history** — sparkline of the last 100 vision readings.
The dashed amber line shows the exit threshold. Vote pips appear below,
filling as consecutive frames agree.

**Motor control** — manual override panel:
- Circular gauge showing current PWM duty
- Slow / Med / Fast preset buttons
- ⏹ Stop — emergency stop
- Speed slider (0–255) with 100ms debounce
- CW / CCW direction toggle
- **↩ Auto** — releases manual override back to orchestrator

**Current node** — name, type, backend (YOLO/REG), cooked% bar, time bar
(for TIMED nodes), and the prompt text.

**Recipe timeline** — all nodes with status icons:
- `·` pending  `▶` active  `✓` done  `!` fallback

**Done overlay** — full-screen completion panel with recipe name, total time,
and a dismiss button.

---

## Recipe YAML format

```yaml
recipe: my_recipe_name

nodes:

  # Timed node — run motor for a fixed duration
  - id: heat_pan
    type: TIMED
    prompt: "Operator instruction shown before this node runs."
    duration_s: 90
    motor:
      speed: 0          # PWM 0-255
      direction: 0      # 0=CCW  1=CW
      stir_s: 90        # seconds motor runs per cycle
      rest_s: 0         # seconds motor rests per cycle

  # Vision node — run until model says "done"
  - id: garlic_golden
    type: VISION
    prompt: "Add garlic and chilli flakes to the oil."
    model: models/garlic_golden_yolo.pt    # path to .pt file
    model_type: yolo                       # "yolo" | "regression"
    exit_threshold: 0.80   # exit when model output ≥ 80%
    frame_votes: 5         # consecutive agreeing frames required
    fallback_s: 240        # hard timeout — node exits regardless
    motor:
      speed: 80
      direction: 1
      stir_s: 3
      rest_s: 9            # vision inference runs during rest window
```

`model_type` is also auto-detected from the filename:
- `*_yolo.pt` → YOLO backend
- `*.pt` → regression backend

---

## Fine-tuning parameters

All parameters are per-node in the YAML — no code changes needed.

| Parameter | Effect |
|---|---|
| `exit_threshold` | Raise it to wait longer (garlic: 0.80, pasta: 0.88) |
| `frame_votes` | More votes = less chance of a false exit (5-10 typical) |
| `fallback_s` | Hard timeout — set generously (2× your expected cook time) |
| `stir_s` | Longer stir = more aggressive mixing |
| `rest_s` | Longer rest = more vision frames per cycle |
| `speed` | Higher PWM = faster spatula |
| `direction` | Swap if spatula moves the wrong way |

---

## Project structure

```
cooki/
├── main.py                  entry point
├── orchestrator.py          recipe execution engine
├── state.py                 thread-safe live state singleton
├── vision.py                unified vision backends (regression + YOLO)
├── trainer.py               live camera capture + regression training
├── yolo_integration.py      image-folder YOLO training
├── motor.py                 Arduino serial bridge
├── ui_server.py             Flask dashboard server
├── cooki_motor.ino          Arduino firmware
├── requirements.txt
│
├── recipes/
│   └── pasta.yaml           one-pot pasta recipe
│
├── templates/
│   └── dashboard.html       web control dashboard
│
├── models/                  trained .pt files (created by trainer tools)
│   ├── garlic_golden_yolo.pt
│   ├── tomato_paste_reduced.pt
│   └── pasta_done_yolo.pt
│
├── training_data/           regression trainer data (trainer.py)
│   └── garlic_golden/
│       ├── frames.csv
│       ├── frame_0000.jpg
│       └── …
│
├── training_images/         YOLO image folders (yolo_integration.py)
│   └── garlic_golden/
│       ├── 0_raw/
│       ├── 25_light/
│       ├── 50_half/
│       ├── 75_mostly/
│       └── 100_done/
│
└── datasets/                auto-generated YOLO dataset (do not edit)
```

---

## Architecture overview

```
Recipe YAML
    │
    ▼
Orchestrator ──────────────────────────────────────────────┐
    │                                                       │
    ├── TIMED node                                          │
    │     Motor (stir/rest cycles)                          ▼
    │     Timer countdown                             State singleton
    │                                                 (thread-safe)
    └── VISION node                                        │ │
          Motor (stir/rest cycles)                         │ │
          Vision inference (during rest)    ───────────────┘ │
            ├── RegressionVisionModel                         │
            │     MobileNetV3-Small                           │
            │     trainer.py → models/*.pt                    ▼
            └── YOLOClassifier                        Flask UI server
                  YOLOv8n-cls                         /stream  MJPEG
                  yolo_integration.py → *_yolo.pt     /events  SSE
                                                      /api/*   motor + confirm
                                                            │
                                                            ▼
                                                      dashboard.html
                                                      (phone / browser)
```

---

## Motor profiles reference

| Name | speed | stir_s | rest_s | Use for |
|---|---|---|---|---|
| Hold | 0 | — | — | Heat only, no stir |
| Gentle fold | 80 | 3 | 9 | Garlic, eggs, delicate |
| Stir-reduce | 150 | 5 | 5 | Sauces, paste reduction |
| Micro-pulse | 110 | 6 | 30 | Long simmers, pasta |
| Aggressive | 220 | 5 | 3 | Thick resistant sauces |

---

## Quick-start checklist

```
□ Flash cooki_motor.ino to Arduino
□ Wire motor driver (D9=PWM, D8=DIR)
□ pip install -r requirements.txt
□ Collect training images for each VISION node
□ python yolo_integration.py --node garlic_golden --images training_images/garlic_golden
□ python yolo_integration.py --node pasta_done    --images training_images/pasta_done
□ python main.py recipes/pasta.yaml --port /dev/ttyACM0
□ Open http://[pi-ip]:5000 on your phone
□ Cook
```