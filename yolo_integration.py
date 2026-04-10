"""
yolo_integration.py — COOKi OS  |  YOLO image-folder trainer

MUCH simpler than live-capture training.
Collect photos on your phone or download from Google Images,
sort into percentage-labeled folders, then train in one command.

FOLDER STRUCTURE
─────────────────
training_images/garlic_golden/
  0_raw/        ← photos of raw garlic  (0% cooked)
  25_light/     ← just starting to turn translucent
  50_half/      ← halfway golden
  75_mostly/    ← mostly golden
  100_done/     ← fully golden-brown  (stop here — any darker = burnt)

• Each subfolder name MUST start with the percentage number.
• Images can be JPG, PNG, BMP, or WebP.
• Aim for 20-50 images per class.
• Mix images taken from different angles, lighting, pan fill levels.

TRAIN A NODE
────────────
  python yolo_integration.py \\
      --node  garlic_golden \\
      --images training_images/garlic_golden

Saves model to:  models/garlic_golden_yolo.pt

In the recipe YAML, reference it with:
  model: models/garlic_golden_yolo.pt
  model_type: yolo          ← or omit — auto-detected from _yolo.pt suffix

HOW INFERENCE WORKS
────────────────────
YOLOv8n-cls outputs a probability per class.
We compute a weighted average:  Σ (class_pct × class_prob)
So "25% likely 50_half + 75% likely 75_mostly" → 0.25×0.5 + 0.75×0.75 = 0.6875
This gives smooth float output in [0, 1] — same interface as the regression model.
"""

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


# ── Dataset preparation ───────────────────────────────────────────────────────

def _prepare_dataset(images_dir: Path, node_id: str) -> Path:
    """
    Scans images_dir for class subfolders (e.g. "50_half"),
    does an 80/20 train/val split, and writes the YOLO classification
    dataset layout that ultralytics expects.

    datasets/{node_id}/
        train/
          0_raw/   25_light/   50_half/   …
        val/
          0_raw/   25_light/   50_half/   …
    """
    import random

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset_root = Path(f"datasets/{node_id}")
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(
            f"No class subfolders found in {images_dir}.\n"
            "Create subfolders named like: 0_raw  25_light  50_half  75_mostly  100_done"
        )

    total = 0
    for cls_dir in subdirs:
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in EXTENSIONS]
        if not images:
            print(f"  [WARN] No images in {cls_dir.name} — skipping")
            continue

        random.shuffle(images)
        split      = max(1, int(len(images) * 0.8))
        train_imgs = images[:split]
        val_imgs   = images[split:] if len(images) > split else images[:1]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = dataset_root / split_name / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy(img, dest / img.name)

        total += len(images)
        print(f"  {cls_dir.name:20s}  {len(train_imgs)} train  {len(val_imgs)} val")

    if total == 0:
        raise ValueError("No images found in any class subfolder.")

    print(f"  Dataset ready: {total} images → {dataset_root}")
    return dataset_root


# ── Training ──────────────────────────────────────────────────────────────────

def train_from_images(node_id:    str,
                      images_dir: str,
                      epochs:     int = 50,
                      imgsz:      int = 224,
                      device:     str = "cpu") -> str:
    """
    Train a YOLOv8n-cls model on a labeled image folder.
    Returns the path to the saved model.
    """
    from ultralytics import YOLO

    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print(f"\n[YOLO TRAIN] Node:   {node_id}")
    print(f"[YOLO TRAIN] Images: {images_path}")
    print(f"[YOLO TRAIN] Epochs: {epochs}  imgsz: {imgsz}  device: {device}\n")

    dataset_path = _prepare_dataset(images_path, node_id)

    model = YOLO("yolov8n-cls.pt")    # downloads ~6 MB on first run
    train_results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=8,
        workers=0,
        project="yolo_runs",
        name=node_id,
        exist_ok=True,
        verbose=False,
    )

    save_dir = None
    if train_results is not None and getattr(train_results, "save_dir", None):
        save_dir = Path(str(train_results.save_dir))
    elif getattr(model, "trainer", None) and getattr(model.trainer, "save_dir", None):
        save_dir = Path(str(model.trainer.save_dir))

    candidates = []
    if save_dir is not None:
        candidates.append(save_dir / "weights" / "best.pt")

    # Fallbacks for different Ultralytics save-dir layouts.
    candidates.extend([
        Path(f"runs/classify/yolo_runs/{node_id}/weights/best.pt"),
        Path(f"yolo_runs/{node_id}/weights/best.pt"),
    ])

    best = next((p for p in candidates if p.exists()), None)
    if best is None:
        discovered = sorted(Path("runs").glob(f"**/yolo_runs/{node_id}/weights/best.pt"))
        best = discovered[0] if discovered else None

    if best is None or not best.exists():
        raise RuntimeError("Training complete but best.pt not found.")

    Path("models").mkdir(exist_ok=True)
    out = Path(f"models/{node_id}_yolo.pt")
    shutil.copy(best, out)
    print(f"\n[YOLO TRAIN] ✓ Saved → {out}")

    _sanity_check(str(out), dataset_path)
    return str(out)


def _sanity_check(model_path: str, dataset_path: Path):
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Rebuild class_pcts from saved model
    class_pcts = {}
    for idx, name in model.names.items():
        m = re.match(r"^(\d+)", str(name))
        class_pcts[idx] = int(m.group(1)) / 100.0 if m else 0.5

    val_dir = dataset_path / "val"
    print("\n[SANITY] Predicted vs actual on val images:")
    checked = 0
    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        m_cls = re.match(r"^(\d+)", cls_dir.name)
        actual_pct = int(m_cls.group(1)) if m_cls else "?"
        for img_path in list(cls_dir.iterdir())[:2]:
            frame   = cv2.imread(str(img_path))
            if frame is None:
                continue
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb, verbose=False)
            probs   = results[0].probs.data.cpu().numpy()
            weighted = sum(class_pcts.get(i, 0.5) * p
                           for i, p in enumerate(probs))
            pred_pct = int(np.clip(weighted, 0, 1) * 100)
            print(f"  pred={pred_pct:3d}%   actual={actual_pct}%   [{img_path.name}]")
            checked += 1
            if checked >= 10:
                return


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COOKi YOLO trainer — train from labeled image folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yolo_integration.py --node garlic_golden --images training_images/garlic_golden
  python yolo_integration.py --node pasta_done    --images training_images/pasta_done --epochs 80
  python yolo_integration.py --node garlic_golden --images training_images/garlic_golden --device cuda

Image folder layout:
  training_images/garlic_golden/
    0_raw/        (20+ photos of raw garlic)
    25_light/     (photos of lightly cooked garlic)
    50_half/      (halfway done)
    75_mostly/    (nearly golden)
    100_done/     (fully golden-brown)
        """,
    )
    parser.add_argument("--node",   required=True, help="Node ID, e.g. garlic_golden")
    parser.add_argument("--images", required=True, help="Path to labeled image folder")
    parser.add_argument("--epochs", type=int,      default=50)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    train_from_images(
        node_id=args.node,
        images_dir=args.images,
        epochs=args.epochs,
        device=args.device,
    )
