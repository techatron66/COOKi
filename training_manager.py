"""
training_manager.py — COOKi OS  |  Training backend

Manages:
  - Node registry  (which nodes exist, what models they have)
  - YOLO training jobs (image-folder based)
  - Camera capture sessions (active learning via browser)
  - Training progress streamed via SSE

Node data lives in  nodes_registry.yaml
YOLO images live in training_images/{node_id}/{pct_label}/
Camera frames live in training_data/{node_id}/
Models saved to     models/
"""

import base64
import csv
import io
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ── YOLO % bins ───────────────────────────────────────────────────────────────
YOLO_BINS = ["0_raw", "25_light", "50_half", "75_mostly", "100_done"]
YOLO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ─────────────────────────────────────────────────────────────────────
REGISTRY_FILE   = Path("nodes_registry.yaml")
TRAINING_IMAGES = Path("training_images")
TRAINING_DATA   = Path("training_data")
MODELS_DIR      = Path("models")


# ── Node registry ─────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_registry(reg: dict):
    with open(REGISTRY_FILE, "w") as f:
        yaml.dump(reg, f, default_flow_style=False)


def list_nodes() -> list[dict]:
    """Return all known nodes with their training status."""
    reg = _load_registry()
    nodes = []
    for node_id, meta in reg.items():
        yolo_path = MODELS_DIR / f"{node_id}_yolo.pt"
        reg_path  = MODELS_DIR / f"{node_id}.pt"

        yolo_images = _count_yolo_images(node_id)
        cam_frames  = _count_cam_frames(node_id)

        nodes.append({
            "id":              node_id,
            "label":           meta.get("label", node_id.replace("_", " ")),
            "description":     meta.get("description", ""),
            "has_yolo_model":  yolo_path.exists(),
            "has_reg_model":   reg_path.exists(),
            "yolo_image_count": yolo_images,
            "cam_frame_count":  cam_frames,
            "yolo_bins":       _yolo_bin_counts(node_id),
        })
    return nodes


def add_node(node_id: str, label: str, description: str = "") -> dict:
    """Register a new node. Creates image + data directories."""
    node_id = node_id.strip().lower().replace(" ", "_")
    if not node_id:
        raise ValueError("node_id cannot be empty")

    reg = _load_registry()
    if node_id in reg:
        raise ValueError(f"Node '{node_id}' already exists")

    reg[node_id] = {"label": label, "description": description}
    _save_registry(reg)

    # Create directory structure
    for bin_name in YOLO_BINS:
        (TRAINING_IMAGES / node_id / bin_name).mkdir(parents=True, exist_ok=True)
    (TRAINING_DATA / node_id).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    return {"ok": True, "node_id": node_id}


def delete_node(node_id: str) -> dict:
    reg = _load_registry()
    reg.pop(node_id, None)
    _save_registry(reg)
    return {"ok": True}


def _count_yolo_images(node_id: str) -> int:
    total = 0
    base  = TRAINING_IMAGES / node_id
    if not base.exists():
        return 0
    for d in base.iterdir():
        if d.is_dir():
            total += sum(1 for f in d.iterdir()
                         if f.suffix.lower() in YOLO_EXTENSIONS)
    return total


def _yolo_bin_counts(node_id: str) -> dict:
    base   = TRAINING_IMAGES / node_id
    counts = {}
    for b in YOLO_BINS:
        d = base / b
        counts[b] = (
            sum(1 for f in d.iterdir() if f.suffix.lower() in YOLO_EXTENSIONS)
            if d.exists() else 0
        )
    return counts


def _count_cam_frames(node_id: str) -> int:
    csv_path = TRAINING_DATA / node_id / "frames.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f))


# ── YOLO image upload ─────────────────────────────────────────────────────────

def save_yolo_image(node_id: str, bin_name: str,
                    file_bytes: bytes, filename: str) -> dict:
    """Save an uploaded image into the correct % bin."""
    if bin_name not in YOLO_BINS:
        raise ValueError(f"Invalid bin: {bin_name}")

    dest_dir = TRAINING_IMAGES / node_id / bin_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filename
    safe = Path(filename).name
    if Path(safe).suffix.lower() not in YOLO_EXTENSIONS:
        safe += ".jpg"

    # Avoid collisions
    dest = dest_dir / safe
    if dest.exists():
        stem = Path(safe).stem
        ext  = Path(safe).suffix
        dest = dest_dir / f"{stem}_{int(time.time()*1000)}{ext}"

    dest.write_bytes(file_bytes)
    return {
        "ok":       True,
        "saved_to": str(dest),
        "bin":      bin_name,
        "count":    _count_yolo_images(node_id),
    }


def delete_yolo_image(node_id: str, bin_name: str, filename: str) -> dict:
    path = TRAINING_IMAGES / node_id / bin_name / filename
    if path.exists():
        path.unlink()
    return {"ok": True, "count": _count_yolo_images(node_id)}


def list_yolo_images(node_id: str, bin_name: str) -> list[str]:
    d = TRAINING_IMAGES / node_id / bin_name
    if not d.exists():
        return []
    return [f.name for f in sorted(d.iterdir())
            if f.suffix.lower() in YOLO_EXTENSIONS]


def get_yolo_thumb(node_id: str, bin_name: str, filename: str) -> Optional[bytes]:
    path = TRAINING_IMAGES / node_id / bin_name / filename
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (120, 90))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


# ── Camera capture (active learning) ─────────────────────────────────────────

class CameraCapture:
    """
    Manages a live camera capture session for one node.
    Frames come in via save_camera_frame() (called from Flask endpoint).
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self.active   = False
        self.node_id  = ""
        self.label    = 0.0
        self.count    = 0
        self._csv_fh  = None
        self._writer  = None

    def start(self, node_id: str) -> dict:
        with self._lock:
            if self.active:
                return {"ok": False, "error": "Capture session already active"}
            self.node_id = node_id
            self.label   = 0.0

            data_dir = TRAINING_DATA / node_id
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / "frames.csv"

            # Count existing
            self.count = _count_cam_frames(node_id)

            write_header = not csv_path.exists() or self.count == 0
            self._csv_fh = open(csv_path, "a", newline="")
            self._writer = csv.DictWriter(
                self._csv_fh, fieldnames=["filename", "label"]
            )
            if write_header:
                self._writer.writeheader()

            self.active = True
            return {"ok": True, "node_id": node_id, "existing_frames": self.count}

    def stop(self) -> dict:
        with self._lock:
            if self._csv_fh:
                self._csv_fh.close()
                self._csv_fh = None
                self._writer = None
            self.active  = False
            count        = self.count
            self.count   = 0
            return {"ok": True, "total_frames": count}

    def set_label(self, label: float):
        with self._lock:
            self.label = max(0.0, min(1.0, label))

    def save_frame(self, node_id: str, jpeg_bytes: bytes) -> dict:
        """Save a JPEG frame (sent from browser) at current label."""
        with self._lock:
            if not self.active or self.node_id != node_id:
                return {"ok": False, "error": "No active session for this node"}

            fname    = f"frame_{self.count:04d}.jpg"
            out_path = TRAINING_DATA / node_id / fname
            out_path.write_bytes(jpeg_bytes)

            self._writer.writerow({
                "filename": fname,
                "label":    round(self.label, 3),
            })
            self._csv_fh.flush()
            self.count += 1

            return {
                "ok":    True,
                "saved": fname,
                "label": round(self.label * 100),
                "total": self.count,
            }

    def status(self) -> dict:
        with self._lock:
            return {
                "active":   self.active,
                "node_id":  self.node_id,
                "label":    round(self.label * 100),
                "count":    self.count,
            }


camera_capture = CameraCapture()


# ── Training job manager ──────────────────────────────────────────────────────

class TrainingJob:
    """Runs a single training job in a background thread, streaming log lines."""

    def __init__(self):
        self._lock    = threading.Lock()
        self.running  = False
        self.node_id  = ""
        self.method   = ""        # "yolo" | "regression"
        self.log:     list[str] = []
        self.progress = 0         # 0-100
        self.error    = ""
        self.done     = False
        self.success  = False

    def start_yolo(self, node_id: str, epochs: int = 50) -> dict:
        with self._lock:
            if self.running:
                return {"ok": False, "error": "A training job is already running"}
            self.running  = True
            self.node_id  = node_id
            self.method   = "yolo"
            self.log      = []
            self.progress = 0
            self.error    = ""
            self.done     = False
            self.success  = False

        t = threading.Thread(
            target=self._run_yolo, args=(node_id, epochs), daemon=True
        )
        t.start()
        return {"ok": True, "node_id": node_id, "method": "yolo"}

    def start_regression(self, node_id: str, epochs: int = 30) -> dict:
        with self._lock:
            if self.running:
                return {"ok": False, "error": "A training job is already running"}
            self.running  = True
            self.node_id  = node_id
            self.method   = "regression"
            self.log      = []
            self.progress = 0
            self.error    = ""
            self.done     = False
            self.success  = False

        t = threading.Thread(
            target=self._run_regression, args=(node_id, epochs), daemon=True
        )
        t.start()
        return {"ok": True, "node_id": node_id, "method": "regression"}

    def _log(self, msg: str):
        with self._lock:
            self.log.append(msg)
            if len(self.log) > 400:
                self.log.pop(0)

    def _set_progress(self, pct: int):
        with self._lock:
            self.progress = pct

    def _run_yolo(self, node_id: str, epochs: int):
        try:
            self._log(f"[YOLO] Starting training for node: {node_id}")
            self._log(f"[YOLO] Epochs: {epochs}")

            import random, shutil as sh
            from pathlib import Path as P

            images_dir   = TRAINING_IMAGES / node_id
            dataset_root = P(f"datasets/{node_id}")

            # Build dataset
            self._log("[YOLO] Preparing dataset from image folders...")
            self._set_progress(5)

            if dataset_root.exists():
                sh.rmtree(dataset_root)

            subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
            if not subdirs:
                raise ValueError(
                    "No image subfolders found. "
                    "Upload images into the % bins first."
                )

            total = 0
            for cls_dir in subdirs:
                imgs = [f for f in cls_dir.iterdir()
                        if f.suffix.lower() in YOLO_EXTENSIONS]
                if not imgs:
                    self._log(f"  [WARN] {cls_dir.name}: no images — skipping")
                    continue
                random.shuffle(imgs)
                split      = max(1, int(len(imgs) * 0.8))
                train_imgs = imgs[:split]
                val_imgs   = imgs[split:] if len(imgs) > split else imgs[:1]

                for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
                    dest = dataset_root / split_name / cls_dir.name
                    dest.mkdir(parents=True, exist_ok=True)
                    for img in split_imgs:
                        sh.copy(img, dest / img.name)

                total += len(imgs)
                self._log(f"  {cls_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val")

            self._log(f"[YOLO] Dataset ready: {total} images")
            self._set_progress(15)

            # Train
            self._log("[YOLO] Loading YOLOv8n-cls base weights...")
            from ultralytics import YOLO
            model = YOLO("yolov8n-cls.pt")
            self._set_progress(20)

            self._log(f"[YOLO] Training for {epochs} epochs...")

            # Patch YOLO's print to capture log lines
            import io as _io, sys as _sys

            class _Tee:
                def __init__(self, outer, real):
                    self.outer = outer
                    self.real  = real
                def write(self, s):
                    if s.strip():
                        self.outer._log(s.rstrip())
                    self.real.write(s)
                def flush(self):
                    self.real.flush()

            old_stdout = _sys.stdout
            _sys.stdout = _Tee(self, old_stdout)

            try:
                model.train(
                    data=str(dataset_root),
                    epochs=epochs,
                    imgsz=224,
                    device="cpu",
                    batch=8,
                    workers=0,
                    project="yolo_runs",
                    name=node_id,
                    exist_ok=True,
                    verbose=True,
                )
            finally:
                _sys.stdout = old_stdout

            self._set_progress(90)

            best = P(f"yolo_runs/{node_id}/weights/best.pt")
            if not best.exists():
                raise RuntimeError("best.pt not found after training")

            MODELS_DIR.mkdir(exist_ok=True)
            out = MODELS_DIR / f"{node_id}_yolo.pt"
            sh.copy(best, out)

            self._log(f"[YOLO] ✓ Model saved → {out}")
            self._set_progress(100)

            with self._lock:
                self.success = True
                self.done    = True
                self.running = False

        except Exception as e:
            self._log(f"[ERROR] {e}")
            with self._lock:
                self.error   = str(e)
                self.done    = True
                self.running = False

    def _run_regression(self, node_id: str, epochs: int):
        try:
            self._log(f"[REG] Starting regression training for: {node_id}")

            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from PIL import Image as PILImage

            csv_path = TRAINING_DATA / node_id / "frames.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    "No camera frames found. "
                    "Run a capture session first."
                )

            samples = []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    p = TRAINING_DATA / node_id / row["filename"]
                    if p.exists():
                        samples.append((str(p), float(row["label"])))

            self._log(f"[REG] {len(samples)} frames loaded")
            if len(samples) < 5:
                raise ValueError(
                    f"Only {len(samples)} frames — need at least 5. Capture more first."
                )

            self._set_progress(5)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(0.35, 0.35, 0.25),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

            class DS(Dataset):
                def __init__(self, samples, tfm):
                    self.samples = samples
                    self.tfm     = tfm
                def __len__(self):  return len(self.samples)
                def __getitem__(self, i):
                    path, label = self.samples[i]
                    x = self.tfm(PILImage.open(path).convert("RGB"))
                    return x, torch.tensor([label], dtype=torch.float32)

            from vision import build_regression_model
            model   = build_regression_model()
            dl      = DataLoader(DS(samples, transform), batch_size=8,
                                 shuffle=True, num_workers=0)
            loss_fn = nn.MSELoss()

            # Phase 1: head only
            for p in model.features.parameters():
                p.requires_grad = False
            opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

            self._log(f"[REG] Phase 1 — classifier head ({epochs} epochs)")
            for ep in range(epochs):
                model.train()
                total = 0.0
                for x, y in dl:
                    pred = torch.sigmoid(model(x))
                    loss = loss_fn(pred, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item()
                avg = total / max(len(dl), 1)
                self._log(f"  Epoch {ep+1:3d}/{epochs}  loss={avg:.5f}")
                pct = 5 + int((ep + 1) / epochs * 40)
                self._set_progress(pct)

            # Phase 2: full fine-tune
            for p in model.features.parameters():
                p.requires_grad = True
            opt2        = torch.optim.Adam(model.parameters(), lr=1e-4)
            fine_epochs = max(10, epochs // 2)

            self._log(f"[REG] Phase 2 — full fine-tune ({fine_epochs} epochs)")
            for ep in range(fine_epochs):
                model.train()
                total = 0.0
                for x, y in dl:
                    pred = torch.sigmoid(model(x))
                    loss = loss_fn(pred, y)
                    opt2.zero_grad(); loss.backward(); opt2.step()
                    total += loss.item()
                avg = total / max(len(dl), 1)
                self._log(f"  Epoch {ep+1:3d}/{fine_epochs}  loss={avg:.5f}")
                pct = 45 + int((ep + 1) / fine_epochs * 45)
                self._set_progress(pct)

            MODELS_DIR.mkdir(exist_ok=True)
            out = MODELS_DIR / f"{node_id}.pt"
            import torch
            torch.save(model.state_dict(), str(out))
            self._log(f"[REG] ✓ Model saved → {out}")
            self._set_progress(100)

            with self._lock:
                self.success = True
                self.done    = True
                self.running = False

        except Exception as e:
            self._log(f"[ERROR] {e}")
            with self._lock:
                self.error   = str(e)
                self.done    = True
                self.running = False

    def status(self) -> dict:
        with self._lock:
            return {
                "running":  self.running,
                "node_id":  self.node_id,
                "method":   self.method,
                "progress": self.progress,
                "log":      self.log[-60:],
                "done":     self.done,
                "success":  self.success,
                "error":    self.error,
            }

    def log_events(self):
        """SSE generator: streams new log lines as they appear."""
        sent = 0
        while True:
            with self._lock:
                lines   = self.log[sent:]
                done    = self.done
                prog    = self.progress
                success = self.success
                err     = self.error
            for line in lines:
                yield f"data: {json.dumps({'type':'log','line':line,'progress':prog})}\n\n"
                sent += 1
            if done:
                yield f"data: {json.dumps({'type':'done','success':success,'error':err,'progress':100 if success else prog})}\n\n"
                return
            time.sleep(0.3)


# Global singletons
training_job    = TrainingJob()
