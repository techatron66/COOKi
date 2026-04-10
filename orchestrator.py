"""
orchestrator.py — COOKi OS  |  Recipe execution engine

Walks the node list from a YAML recipe, executing each node in sequence.
All runtime state is pushed into state.py so the web UI can display it live.
Operator confirmation waits on state.confirm_event, set by the UI confirm
button OR by pressing ENTER in the terminal (unless --headless).
"""

import threading
import time
import yaml

from motor      import MotorController
from vision     import load_vision_model, Camera
from node       import node_from_dict, TimedNode, VisionNode
from state      import state, NodeMeta


_BAR = "═" * 56


class Orchestrator:
    def __init__(self,
                 recipe_path: str,
                 motor_port:  str  = "/dev/ttyUSB0",
                 mock:        bool = False,
                 headless:    bool = False,
                 cam_index:   int  = 0):

        with open(recipe_path) as f:
            recipe = yaml.safe_load(f)

        self.recipe_name = recipe["recipe"]
        self.nodes       = [node_from_dict(n) for n in recipe["nodes"]]
        self.motor       = MotorController(port=motor_port, mock=mock)
        self.camera      = Camera(index=cam_index)
        self.headless    = headless

        self._vision_cache: dict[str, object] = {}
        self._cam_running = True

        # Populate shared state with recipe metadata
        metas = [NodeMeta(i, n.id, n.type) for i, n in enumerate(self.nodes)]
        state.start_session(self.recipe_name, metas)

        # Background thread — keeps state.latest_frame fresh at ~25fps
        threading.Thread(target=self._camera_loop, daemon=True).start()

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self):
        while self._cam_running:
            frame = self.camera.read()
            if frame is not None:
                state.set_frame(frame)
            time.sleep(0.04)

    # ── Vision cache ──────────────────────────────────────────────────────────

    def _get_vision(self, node: VisionNode):
        if node.model_path not in self._vision_cache:
            self._vision_cache[node.model_path] = load_vision_model(
                node.model_path, node.model_type, node.yolo_bins
            )
        return self._vision_cache[node.model_path]

    # ── Operator prompt ───────────────────────────────────────────────────────

    def _prompt(self, idx: int, node_id: str, prompt: str):
        state.set_awaiting(idx, prompt)

        print(f"\n{_BAR}")
        print(f"  NODE {idx+1}  →  {node_id}")
        print(f"  {prompt}")
        print(_BAR)

        if self.headless:
            print("  [HEADLESS] Waiting for UI confirm button…")
            state.wait_for_confirm()
        else:
            # Accept keyboard ENTER *or* UI button — whichever comes first
            t = threading.Thread(
                target=lambda: (
                    input("  → ENTER when ready  (or tap Confirm in the UI)… "),
                    state.confirm()
                ),
                daemon=True,
            )
            t.start()
            state.wait_for_confirm()

        print(f"  ✓ Confirmed. Running {node_id}.\n")

    # ── Node: TIMED ───────────────────────────────────────────────────────────

    def _run_timed(self, node: TimedNode, idx: int):
        print(f"[TIMED] {node.id}  duration={node.duration_s}s")
        state.set_node_started(idx, duration_s=node.duration_s)
        m     = node.motor
        start = time.time()

        while True:
            elapsed   = time.time() - start
            remaining = node.duration_s - elapsed
            if remaining <= 0:
                break

            if m.speed > 0:
                # Stir phase
                stir_t = min(m.stir_s, remaining)
                self.motor.send(m.speed, m.direction)
                state.update_motor(m.speed, m.direction, True)
                time.sleep(stir_t)
                self.motor.stop()
                state.update_motor(0, m.direction, False)

                # Rest phase
                rest_t = min(m.rest_s, node.duration_s - (time.time() - start))
                if rest_t > 0:
                    time.sleep(rest_t)
            else:
                time.sleep(1)

            elapsed = time.time() - start
            state.update_timed(elapsed, node.duration_s)
            print(f"  {node.id}: {int(elapsed)}s / {int(node.duration_s)}s   ", end="\r")

        self.motor.stop()
        state.update_motor(0, m.direction, False)
        state.set_node_done(idx)
        print(f"\n[TIMED] {node.id} — complete.")

    # ── Node: VISION ──────────────────────────────────────────────────────────

    def _run_vision(self, node: VisionNode, idx: int):
        print(
            f"[VISION/{node.model_type.upper()}] {node.id}  "
            f"exit≥{int(node.exit_threshold*100)}%  "
            f"votes={node.frame_votes}  fallback={node.fallback_s}s"
        )
        state.set_node_started(idx, exit_threshold=node.exit_threshold)
        vision = self._get_vision(node)
        m      = node.motor
        votes  = 0
        start  = time.time()

        while True:
            elapsed = time.time() - start

            # Hard fallback timer
            if elapsed > node.fallback_s:
                print(f"\n  [FALLBACK] {node.id} — timed out at {int(elapsed)}s")
                state.set_node_done(idx, "fallback")
                break

            # ── Stir phase ────────────────────────────────────────────────────
            self.motor.send(m.speed, m.direction)
            state.update_motor(m.speed, m.direction, True)
            time.sleep(m.stir_s)
            self.motor.stop()
            state.update_motor(0, m.direction, False)

            # ── Vision phase (during rest window) ─────────────────────────────
            rest_end = time.time() + m.rest_s
            while time.time() < rest_end:
                frame = state.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                pct     = vision.infer(frame)
                elapsed = time.time() - start
                state.update_vision(pct, votes, node.frame_votes, node.model_type)
                state.update_timed(elapsed, node.fallback_s)

                filled = int(pct * 20)
                bar    = "█" * filled + "░" * (20 - filled)
                print(
                    f"  {node.id}: [{bar}] {int(pct*100):3d}%  "
                    f"votes={votes}/{node.frame_votes}  t={int(elapsed)}s  ",
                    end="\r",
                )

                if pct >= node.exit_threshold:
                    votes += 1
                    state.update_vision(pct, votes, node.frame_votes, node.model_type)
                    if votes >= node.frame_votes:
                        self.motor.stop()
                        state.update_motor(0, m.direction, False)
                        state.set_node_done(idx)
                        print(
                            f"\n[VISION] {node.id} — DONE  "
                            f"{int(pct*100)}%  ({votes} consecutive frames)"
                        )
                        return
                else:
                    votes = 0

                time.sleep(0.15)

        self.motor.stop()

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n{_BAR}")
        print(f"  COOKi OS  ·  {self.recipe_name.upper().replace('_',' ')}")
        print(f"  {len(self.nodes)} nodes  ·  UI → http://localhost:5000")
        print(_BAR)

        for i, node in enumerate(self.nodes):
            print(f"\nNode {i+1}/{len(self.nodes)}  [{node.type}]  {node.id}")
            self._prompt(i, node.id, node.prompt)

            if isinstance(node, TimedNode):
                self._run_timed(node, i)
            elif isinstance(node, VisionNode):
                self._run_vision(node, i)

        self._cam_running = False
        self.motor.close()
        self.camera.release()
        state.set_done()

        print(f"\n{_BAR}")
        print(f"  {self.recipe_name.upper().replace('_',' ')} — COMPLETE  🍝")
        print(f"  Buon appetito.")
        print(_BAR)
