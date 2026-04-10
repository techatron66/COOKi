"""
trainer.py — COOKi OS  |  Live camera trainer  (regression backend)

Captures frames from the Pi/PC camera while you cook, lets you label each
frame with a "% cooked" value via keyboard, then trains a MobileNetV3-Small
regression model on the collected dataset.

CAPTURE MODE (default):
  python trainer.py --node garlic_golden

TRAIN ONLY (skip capture):
  python trainer.py --node garlic_golden --train-only

KEY MAP during capture:
  0-9     Set label  (0=0%  3=30%  7=70%  9=90%)
  [  ]    Nudge label  -5% / +5%
  SPACE   Capture current frame at current label
  T       Stop capture → train immediately
  Q       Quit without training

STRATEGY — aim for 60-120 frames spread across the full cooking spectrum:
  • Start with the ingredient raw (label 0%)
  • Capture every 15-20 seconds as it cooks
  • Label each capture honestly (how cooked does it look?)
  • Capture the "done" moment repeatedly at 80-100%

Training stores frames in:  training_data/{node_id}/frames.csv
Saves model to:             models/{node_id}.pt
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vision import build_regression_model


# ── Dataset ───────────────────────────────────────────────────────────────────

_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CookDataset(Dataset):
    def __init__(self, node_id: str, augment: bool = True):
        self.root      = Path(f"training_data/{node_id}")
        self.transform = _TRANSFORM_TRAIN if augment else _TRANSFORM_VAL
        self.samples: list[tuple[str, float]] = []

        csv_path = self.root / "frames.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No training data at {csv_path}. "
                "Run trainer.py without --train-only first."
            )

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.samples.append(
                    (str(self.root / row["filename"]), float(row["label"]))
                )

        print(f"[DATASET] {len(self.samples)} frames for node '{node_id}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        x   = self.transform(img)
        y   = torch.tensor([label], dtype=torch.float32)
        return x, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(node_id: str, epochs: int = 30, lr: float = 1e-3, batch_size: int = 8):
    ds = CookDataset(node_id, augment=True)
    if len(ds) < 10:
        print(f"[WARN] Only {len(ds)} frames — aim for 60+ for reliable inference.")

    dl      = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model   = build_regression_model()
    loss_fn = nn.MSELoss()

    # Phase 1 — classifier head only (fast warm-up)
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    print(f"\n[TRAIN] Phase 1 — head only  ({epochs} epochs)")
    _train_loop(model, dl, opt, loss_fn, epochs)

    # Phase 2 — full fine-tune at lower lr
    for p in model.features.parameters():
        p.requires_grad = True
    opt2        = torch.optim.Adam(model.parameters(), lr=lr / 10)
    fine_epochs = max(10, epochs // 2)

    print(f"[TRAIN] Phase 2 — full fine-tune  ({fine_epochs} epochs)")
    _train_loop(model, dl, opt2, loss_fn, fine_epochs)

    # Save
    Path("models").mkdir(exist_ok=True)
    out = f"models/{node_id}.pt"
    torch.save(model.state_dict(), out)
    print(f"\n[TRAIN] ✓ Saved → {out}")
    _sanity_check(model, ds)
    return out


def _train_loop(model, dl, opt, loss_fn, epochs):
    model.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in dl:
            pred = torch.sigmoid(model(x))
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(len(dl), 1)
        print(f"  Epoch {ep+1:3d}/{epochs}  loss={avg:.5f}", end="\r")
    print()


def _sanity_check(model, ds: CookDataset):
    model.eval()
    print("\n[SANITY] Predicted vs actual on sample frames:")
    indices = list(range(0, len(ds), max(1, len(ds) // 8)))[:8]
    for i in indices:
        x, y = ds[i]
        with torch.no_grad():
            pred = float(torch.sigmoid(model(x.unsqueeze(0))).squeeze())
        print(f"  pred={int(pred*100):3d}%   actual={int(float(y)*100):3d}%")


def _headless_capture_session(node_id: str, data_dir: Path, csv_path: Path,
                              frame_count: int):
    label        = 0.0
    last_capture  = 0.0
    CAPTURE_COOLDOWN = 0.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["filename", "label"])

    try:
        print("[CAPTURE] Running in terminal mode because OpenCV GUI is unavailable.")
        print("  Commands: 0-9 set label   [ ] nudge ±5%   space capture   t train   q quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Could not read from camera.")
                break

            pct = int(label * 100)
            raw_command = input(f"[CAPTURE] Label={pct}% Frames={frame_count} > ")
            command = raw_command.lower()

            if command == "q":
                break
            if command == "t":
                csv_file.close()
                cap.release()
                if frame_count < 5:
                    print("[ERROR] Need at least 5 frames. Capture more first.")
                    return
                print(f"\n[CAPTURE] Finished. Training on {frame_count} frames…")
                train(node_id)
                return
            if command in {"[", "]"}:
                delta = -0.05 if command == "[" else 0.05
                label = min(1.0, max(0.0, round(label + delta, 2)))
                continue
            if len(command) == 1 and command.isdigit():
                label = int(command) / 10.0
                continue
            if command not in {"", " ", "space"}:
                print("[INFO] Unknown command.")
                continue

            now = time.time()
            if now - last_capture < CAPTURE_COOLDOWN:
                continue

            fname = f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(data_dir / fname), frame)
            writer.writerow({"filename": fname, "label": round(label, 3)})
            csv_file.flush()
            frame_count += 1
            last_capture = now
            print(f"  ✓ {fname}  label={pct}%  total={frame_count}")
    finally:
        csv_file.close()
        cap.release()



# ── Capture UI ────────────────────────────────────────────────────────────────

def capture_session(node_id: str):
    data_dir = Path(f"training_data/{node_id}")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "frames.csv"

    # Count existing frames
    existing = []
    if csv_path.exists():
        with open(csv_path) as f:
            existing = list(csv.DictReader(f))
    frame_count = len(existing)

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        _headless_capture_session(node_id, data_dir, csv_path, frame_count)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    label            = 0.0
    last_capture     = 0.0
    CAPTURE_COOLDOWN = 0.5
    flash            = 0

    print(f"\n[CAPTURE] Node: {node_id}  |  {frame_count} existing frames")
    print("  Keys:  0-9 set label   [ ] nudge ±5%   SPACE capture   T train   Q quit")

    write_header = not csv_path.exists() or frame_count == 0
    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["filename", "label"])
    if write_header:
        writer.writeheader()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        pct     = int(label * 100)

        # Capture flash
        if flash > 0:
            white   = display.copy()
            white[:] = (255, 255, 255)
            display  = cv2.addWeighted(display, 1 - flash / 8, white, flash / 8, 0)
            flash   -= 1

        # Label bar at bottom
        bw = int(display.shape[1] * label)
        cv2.rectangle(display, (0, display.shape[0] - 8),
                      (bw, display.shape[0]), (60, 200, 100), -1)

        # HUD text
        cv2.putText(display, f"Label: {pct}%", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 210, 100), 2, cv2.LINE_AA)
        cv2.putText(display, f"Frames: {frame_count}", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display,
                    "SPACE=capture   [  ]=nudge   T=train   Q=quit",
                    (12, display.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(f"COOKi Trainer — {node_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("9"):
            label = (key - ord("0")) / 10.0
        elif key == ord("["):
            label = max(0.0, round(label - 0.05, 2))
        elif key == ord("]"):
            label = min(1.0, round(label + 0.05, 2))
        elif key == ord(" "):
            now = time.time()
            if now - last_capture >= CAPTURE_COOLDOWN:
                fname = f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(data_dir / fname), frame)
                writer.writerow({"filename": fname, "label": round(label, 3)})
                csv_file.flush()
                frame_count  += 1
                last_capture  = now
                flash         = 8
                print(f"  ✓ {fname}  label={pct}%  total={frame_count}")
        elif key == ord("t"):
            csv_file.close()
            cap.release()
            cv2.destroyAllWindows()
            if frame_count < 5:
                print("[ERROR] Need at least 5 frames. Capture more first.")
                return
            print(f"\n[CAPTURE] Finished. Training on {frame_count} frames…")
            train(node_id)
            return

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[CAPTURE] Session ended. {frame_count} frames saved.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COOKi live camera trainer (regression backend)",
    )
    parser.add_argument("--node",       required=True,
                        help="Node ID, e.g.  garlic_golden")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip capture — retrain on existing frames")
    parser.add_argument("--epochs",     type=int, default=30)
    args = parser.parse_args()

    if args.train_only:
        train(args.node, epochs=args.epochs)
    else:
        capture_session(args.node)
