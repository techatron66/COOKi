"""
ui_server.py — COOKi OS  |  Web dashboard + Training server

Cook dashboard:   http://[pi-ip]:5000/
Training studio:  http://[pi-ip]:5000/train

New training endpoints:
  GET  /train                        training studio HTML
  GET  /api/nodes                    list all nodes + training status
  POST /api/nodes                    add new node  {id, label, description}
  DELETE /api/nodes/<id>             remove node from registry
  GET  /api/nodes/<id>/yolo/images/<bin>   list images in a bin
  POST /api/nodes/<id>/yolo/upload/<bin>   upload images (multipart)
  DELETE /api/nodes/<id>/yolo/<bin>/<file> delete one image
  GET  /api/nodes/<id>/yolo/thumb/<bin>/<file>  thumbnail JPEG
  POST /api/train/yolo/<id>          start YOLO training job
  POST /api/train/regression/<id>    start regression training job
  GET  /api/train/events             SSE stream of training log
  GET  /api/train/status             training job status JSON
  POST /api/capture/start            start camera capture session
  POST /api/capture/stop             stop capture session
  POST /api/capture/label            set label  {node_id, label}
  POST /api/capture/frame            save one frame (JPEG body)
  GET  /api/capture/status           capture session status
"""

import json
import threading
import time

import cv2
import numpy as np
from flask import (Flask, Response, render_template,
                   request, jsonify, send_file)
import io

from state import state
from training_manager import (
    list_nodes, add_node, delete_node,
    save_yolo_image, delete_yolo_image, list_yolo_images, get_yolo_thumb,
    camera_capture, training_job,
    YOLO_BINS,
)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"]    = 64 * 1024 * 1024   # 64 MB upload limit


# ── MJPEG camera stream ───────────────────────────────────────────────────────

def _blank_frame():
    f = np.zeros((480, 640, 3), dtype="uint8")
    cv2.putText(f, "No camera signal", (170, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 70, 70), 2)
    return f


def _gen_frames(overlay: bool = True):
    blank = _blank_frame()
    while True:
        frame = state.get_frame()
        if frame is None:
            frame = blank.copy()
        else:
            frame = frame.copy()

        if overlay:
            s = state.to_dict()
            if s["status"] == "cooking":
                pct = s["cooked_pct"]
                ov  = frame.copy()
                cv2.rectangle(ov, (8, 8), (100, 50), (15, 15, 15), -1)
                frame = cv2.addWeighted(ov, 0.65, frame, 0.35, 0)
                cv2.putText(frame, f"{pct:.0f}%", (14, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (70, 200, 100), 2)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.04)


def _gen_train_frames():
    """Clean camera stream for the training page (no cook overlay)."""
    yield from _gen_frames(overlay=False)


# ── SSE state stream ──────────────────────────────────────────────────────────

def _gen_events():
    while True:
        yield f"data: {json.dumps(state.to_dict())}\n\n"
        time.sleep(0.3)


# ══════════════════════════════════════════════════════════════════════════════
# COOK DASHBOARD ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    return Response(_gen_events(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/state")
def api_state():
    return jsonify(state.to_dict())


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    state.confirm()
    return jsonify({"ok": True})


@app.route("/api/motor", methods=["POST"])
def api_motor():
    data      = request.get_json(force=True) or {}
    speed     = max(0, min(255, int(data.get("speed", 0))))
    direction = 1 if data.get("direction", 1) else 0
    with state._lock:
        state.motor_speed     = speed
        state.motor_direction = direction
        state.motor_override  = True
    return jsonify({"ok": True, "speed": speed, "direction": direction})


@app.route("/api/motor/stop", methods=["POST"])
def api_motor_stop():
    with state._lock:
        state.motor_speed    = 0
        state.motor_override = True
    return jsonify({"ok": True})


@app.route("/api/motor/release", methods=["POST"])
def api_motor_release():
    with state._lock:
        state.motor_override = False
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING STUDIO ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/train")
def training_studio():
    return render_template("train.html")


@app.route("/train/stream")
def train_stream():
    return Response(_gen_train_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Node registry ─────────────────────────────────────────────────────────────

@app.route("/api/nodes", methods=["GET"])
def api_nodes_list():
    return jsonify(list_nodes())


@app.route("/api/nodes", methods=["POST"])
def api_nodes_add():
    data = request.get_json(force=True) or {}
    try:
        result = add_node(
            node_id=data.get("id", ""),
            label=data.get("label", ""),
            description=data.get("description", ""),
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/nodes/<node_id>", methods=["DELETE"])
def api_nodes_delete(node_id):
    return jsonify(delete_node(node_id))


# ── YOLO image management ─────────────────────────────────────────────────────

@app.route("/api/nodes/<node_id>/yolo/bins")
def api_yolo_bins(node_id):
    return jsonify({"bins": YOLO_BINS})


@app.route("/api/nodes/<node_id>/yolo/images/<bin_name>")
def api_yolo_images(node_id, bin_name):
    return jsonify({"images": list_yolo_images(node_id, bin_name)})


@app.route("/api/nodes/<node_id>/yolo/upload/<bin_name>", methods=["POST"])
def api_yolo_upload(node_id, bin_name):
    files   = request.files.getlist("images")
    results = []
    for f in files:
        try:
            res = save_yolo_image(node_id, bin_name,
                                  f.read(), f.filename or "upload.jpg")
            results.append(res)
        except Exception as e:
            results.append({"ok": False, "error": str(e),
                            "filename": f.filename})
    return jsonify({"results": results, "uploaded": len([r for r in results if r.get("ok")])})


@app.route("/api/nodes/<node_id>/yolo/<bin_name>/<filename>", methods=["DELETE"])
def api_yolo_delete(node_id, bin_name, filename):
    return jsonify(delete_yolo_image(node_id, bin_name, filename))


@app.route("/api/nodes/<node_id>/yolo/thumb/<bin_name>/<filename>")
def api_yolo_thumb(node_id, bin_name, filename):
    data = get_yolo_thumb(node_id, bin_name, filename)
    if data is None:
        return "", 404
    return Response(data, mimetype="image/jpeg")


# ── Training jobs ─────────────────────────────────────────────────────────────

@app.route("/api/train/yolo/<node_id>", methods=["POST"])
def api_train_yolo(node_id):
    data   = request.get_json(force=True) or {}
    epochs = int(data.get("epochs", 50))
    result = training_job.start_yolo(node_id, epochs)
    return jsonify(result)


@app.route("/api/train/regression/<node_id>", methods=["POST"])
def api_train_regression(node_id):
    data   = request.get_json(force=True) or {}
    epochs = int(data.get("epochs", 30))
    result = training_job.start_regression(node_id, epochs)
    return jsonify(result)


@app.route("/api/train/status")
def api_train_status():
    return jsonify(training_job.status())


@app.route("/api/train/events")
def api_train_events():
    return Response(
        training_job.log_events(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Camera capture ────────────────────────────────────────────────────────────

@app.route("/api/capture/start", methods=["POST"])
def api_capture_start():
    data = request.get_json(force=True) or {}
    return jsonify(camera_capture.start(data.get("node_id", "")))


@app.route("/api/capture/stop", methods=["POST"])
def api_capture_stop():
    return jsonify(camera_capture.stop())


@app.route("/api/capture/label", methods=["POST"])
def api_capture_label():
    data  = request.get_json(force=True) or {}
    label = float(data.get("label", 0)) / 100.0
    camera_capture.set_label(label)
    return jsonify({"ok": True, "label": round(label * 100)})


@app.route("/api/capture/frame", methods=["POST"])
def api_capture_frame():
    node_id = request.args.get("node_id", camera_capture.node_id)
    jpeg    = request.data
    if not jpeg:
        return jsonify({"ok": False, "error": "No image data"}), 400
    return jsonify(camera_capture.save_frame(node_id, jpeg))


@app.route("/api/capture/status")
def api_capture_status():
    return jsonify(camera_capture.status())


# ── Server launcher ───────────────────────────────────────────────────────────

def start_server(host: str = "0.0.0.0", port: int = 5000):
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port,
                               threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"[UI] Cook dashboard  → http://{host}:{port}/")
    print(f"[UI] Train studio    → http://{host}:{port}/train")
    return t
