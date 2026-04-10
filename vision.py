"""
vision.py — COOKi OS  |  Unified vision backends

Two inference backends, same interface:   model.infer(bgr_frame) -> float 0-1

  RegressionVisionModel  — MobileNetV3-Small, single sigmoid output
                           Trained with:  trainer.py  (live camera + slider)

  YOLOClassifier         — YOLOv8n-cls, probability-weighted % cooked
                           Trained with:  yolo_integration.py  (image folders)

load_vision_model(path, model_type) returns whichever backend is appropriate.
"""

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


# ── Transform (shared by regression model) ────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Model builder (also used by trainer.py) ───────────────────────────────────

def build_regression_model() -> nn.Module:
    """MobileNetV3-Small with a single float regression head."""
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m


# ── Regression backend ────────────────────────────────────────────────────────

class RegressionVisionModel:
    """
    Live-trained via trainer.py.
    Inputs:  BGR frame from OpenCV
    Output:  sigmoid(logit) → float  0.0 (raw) – 1.0 (done)
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model  = build_regression_model().to(device)
        p = Path(model_path)
        if p.exists():
            self.model.load_state_dict(
                torch.load(str(p), map_location=device)
            )
            print(f"[VISION] Regression model loaded: {p}")
        else:
            print(f"[VISION] ⚠ No model at {p} — returning 0.0 until trained")
        self.model.eval()

    def infer(self, frame) -> float:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.model(tensor)
        return float(torch.sigmoid(raw).squeeze().cpu())


# ── YOLO backend ──────────────────────────────────────────────────────────────

class YOLOClassifier:
    """
    Trained via yolo_integration.py from image folders.
    Class names must encode the target percentage as a leading number:
      "0_raw", "25_light", "50_half", "75_mostly", "100_done"
    Inference returns a weighted average of class probabilities.
    """
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model      = YOLO(model_path)
        self.class_pcts = self._parse_class_pcts()
        print(f"[VISION] YOLO model loaded: {model_path}")
        print(f"[VISION] YOLO classes: {self.class_pcts}")

    def _parse_class_pcts(self) -> dict[int, float]:
        import re
        result = {}
        for idx, name in self.model.names.items():
            m = re.match(r"^(\d+)", str(name))
            result[idx] = int(m.group(1)) / 100.0 if m else 0.5
        return result

    def infer(self, frame) -> float:
        import numpy as np
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, verbose=False)
        probs   = results[0].probs.data.cpu().numpy()
        weighted = sum(self.class_pcts.get(i, 0.5) * p
                       for i, p in enumerate(probs))
        return float(np.clip(weighted, 0.0, 1.0))


# ── Unified loader ────────────────────────────────────────────────────────────

def load_vision_model(model_path: str,
                       model_type: str = "regression",
                       yolo_bins=None):
    """
    Load the right backend for a recipe node.
    model_type: "regression" | "yolo"
    """
    if model_type == "yolo":
        return YOLOClassifier(model_path)
    return RegressionVisionModel(model_path)


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera index {index} not available. "
                "Check connection or try a different index."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[CAMERA] Opened camera {index}  (640×480)")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
