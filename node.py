"""
node.py — COOKi OS  |  Recipe node schema + YAML parser

Defines typed node objects consumed by orchestrator.py.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MotorProfile:
    speed: int = 0
    direction: int = 1
    stir_s: float = 0.0
    rest_s: float = 0.0


@dataclass
class BaseNode:
    id: str
    type: str
    prompt: str
    motor: MotorProfile


@dataclass
class TimedNode(BaseNode):
    duration_s: float = 0.0


@dataclass
class VisionNode(BaseNode):
    model_path: str = ""
    model_type: str = "regression"
    exit_threshold: float = 0.85
    frame_votes: int = 5
    fallback_s: float = 180.0
    yolo_bins: list[int] | None = None


def _to_motor(data: dict[str, Any]) -> MotorProfile:
    return MotorProfile(
        speed=max(0, min(255, int(data.get("speed", 0)))),
        direction=1 if int(data.get("direction", 1)) else 0,
        stir_s=max(0.0, float(data.get("stir_s", 0.0))),
        rest_s=max(0.0, float(data.get("rest_s", 0.0))),
    )


def node_from_dict(data: dict[str, Any]) -> BaseNode:
    node_type = str(data.get("type", "")).strip().upper()
    node_id = str(data.get("id", "")).strip()
    if not node_id:
        raise ValueError("Node 'id' is required")

    prompt = str(data.get("prompt", "")).strip()
    motor = _to_motor(data.get("motor", {}) or {})

    if node_type == "TIMED":
        if "duration_s" not in data:
            raise ValueError(f"TIMED node '{node_id}' missing 'duration_s'")
        return TimedNode(
            id=node_id,
            type="TIMED",
            prompt=prompt,
            motor=motor,
            duration_s=max(0.0, float(data.get("duration_s", 0))),
        )

    if node_type == "VISION":
        model_path = str(data.get("model", "")).strip()
        model_type = str(data.get("model_type", "")).strip().lower()
        if not model_type:
            model_type = "yolo" if model_path.endswith("_yolo.pt") else "regression"
        if model_type not in {"regression", "yolo"}:
            raise ValueError(
                f"VISION node '{node_id}' has invalid model_type '{model_type}'"
            )

        return VisionNode(
            id=node_id,
            type="VISION",
            prompt=prompt,
            motor=motor,
            model_path=model_path,
            model_type=model_type,
            exit_threshold=float(data.get("exit_threshold", 0.85)),
            frame_votes=max(1, int(data.get("frame_votes", 5))),
            fallback_s=max(1.0, float(data.get("fallback_s", 180))),
            yolo_bins=data.get("yolo_bins"),
        )

    raise ValueError(f"Node '{node_id}' has unknown type '{node_type}'")
