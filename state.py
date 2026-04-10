"""
state.py — COOKi OS  |  Global live state

Single shared object, written by the orchestrator, read by the UI server.
All field access is under an RLock. The UI polls this every 300ms via SSE.
"""

import threading
import time
from typing import Optional
import numpy as np


class NodeMeta:
    def __init__(self, idx: int, node_id: str, node_type: str):
        self.idx       = idx
        self.id        = node_id
        self.type      = node_type          # TIMED | VISION
        self.status    = "pending"          # pending | active | done | fallback
        self.started_at:  Optional[float] = None
        self.finished_at: Optional[float] = None

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict:
        return {
            "idx":     self.idx,
            "id":      self.id,
            "type":    self.type,
            "status":  self.status,
            "elapsed": round(self.elapsed(), 1),
        }


class CookiState:
    def __init__(self):
        self._lock = threading.RLock()

        # Recipe ──────────────────────────────────────────────────────────────
        self.recipe_name:  str = ""
        self.node_metas:   list[NodeMeta] = []
        self.current_idx:  int = -1

        # Vision ──────────────────────────────────────────────────────────────
        self.cooked_pct:          float = 0.0
        self.votes:               int   = 0
        self.votes_needed:        int   = 0
        self.vision_backend:      str   = "regression"   # "regression" | "yolo"
        self.exit_threshold:      float = 0.85
        self.confidence_history:  list[float] = []

        # Motor ───────────────────────────────────────────────────────────────
        self.motor_speed:     int  = 0
        self.motor_direction: int  = 1
        self.is_stirring:     bool = False
        self.motor_override:  bool = False

        # Timers ──────────────────────────────────────────────────────────────
        self.session_start:    Optional[float] = None
        self.node_duration_s:  float = 0.0
        self.node_elapsed_s:   float = 0.0

        # Flow ────────────────────────────────────────────────────────────────
        self.status:  str = "idle"   # idle | awaiting | cooking | done | error
        self.prompt:  str = ""
        self.confirm_event = threading.Event()

        # Camera ──────────────────────────────────────────────────────────────
        self.latest_frame: Optional[np.ndarray] = None

    # ── Write helpers (orchestrator → state) ──────────────────────────────────

    def start_session(self, recipe_name: str, node_metas: list[NodeMeta]):
        with self._lock:
            self.recipe_name  = recipe_name
            self.node_metas   = node_metas
            self.session_start = time.time()
            self.status       = "idle"

    def set_awaiting(self, idx: int, prompt: str):
        with self._lock:
            self.current_idx = idx
            self.status      = "awaiting"
            self.prompt      = prompt
            self.confirm_event.clear()
            if 0 <= idx < len(self.node_metas):
                self.node_metas[idx].status = "active"

    def wait_for_confirm(self):
        self.confirm_event.wait()

    def confirm(self):
        with self._lock:
            self.status = "cooking"
            self.confirm_event.set()

    def set_node_started(self, idx: int, duration_s: float = 0.0,
                         exit_threshold: float = 0.85):
        with self._lock:
            self.status           = "cooking"
            self.node_elapsed_s   = 0.0
            self.node_duration_s  = duration_s
            self.cooked_pct       = 0.0
            self.votes            = 0
            self.exit_threshold   = exit_threshold
            self.confidence_history = []
            if 0 <= idx < len(self.node_metas):
                self.node_metas[idx].started_at = time.time()

    def set_node_done(self, idx: int, status: str = "done"):
        with self._lock:
            if 0 <= idx < len(self.node_metas):
                self.node_metas[idx].status      = status
                self.node_metas[idx].finished_at = time.time()
            self.is_stirring  = False
            self.motor_speed  = 0

    def update_vision(self, pct: float, votes: int, votes_needed: int, backend: str):
        with self._lock:
            self.cooked_pct      = pct
            self.votes           = votes
            self.votes_needed    = votes_needed
            self.vision_backend  = backend
            self.confidence_history.append(round(pct, 3))
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)

    def update_motor(self, speed: int, direction: int, stirring: bool):
        with self._lock:
            self.motor_speed      = speed
            self.motor_direction  = direction
            self.is_stirring      = stirring

    def update_timed(self, elapsed: float, duration: float):
        with self._lock:
            self.node_elapsed_s  = elapsed
            self.node_duration_s = duration
            self.cooked_pct      = min(elapsed / max(duration, 1), 1.0)

    def set_frame(self, frame):
        with self._lock:
            self.latest_frame = frame

    def set_done(self):
        with self._lock:
            self.status       = "done"
            self.is_stirring  = False
            self.motor_speed  = 0

    # ── Read helpers (Flask → state) ──────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            current = None
            if 0 <= self.current_idx < len(self.node_metas):
                current = self.node_metas[self.current_idx].to_dict()

            total_elapsed = (time.time() - self.session_start
                             if self.session_start else 0.0)

            return {
                "recipe":              self.recipe_name,
                "status":              self.status,
                "prompt":              self.prompt,
                "current_idx":         self.current_idx,
                "current_node":        current,
                "nodes":               [n.to_dict() for n in self.node_metas],
                "cooked_pct":          round(self.cooked_pct * 100, 1),
                "votes":               self.votes,
                "votes_needed":        self.votes_needed,
                "exit_threshold":      round(self.exit_threshold * 100, 1),
                "vision_backend":      self.vision_backend,
                "confidence_history":  self.confidence_history[-100:],
                "motor_speed":         self.motor_speed,
                "motor_direction":     self.motor_direction,
                "is_stirring":         self.is_stirring,
                "motor_override":      self.motor_override,
                "node_elapsed_s":      round(self.node_elapsed_s, 1),
                "node_duration_s":     round(self.node_duration_s, 1),
                "total_elapsed_s":     round(total_elapsed, 1),
            }

    def get_frame(self):
        with self._lock:
            return self.latest_frame


# Global singleton ─────────────────────────────────────────────────────────────
state = CookiState()
