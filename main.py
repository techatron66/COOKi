"""
main.py — COOKi OS  |  Entry point

Starts the web dashboard then runs the recipe orchestrator in the same process.
The orchestrator thread emits state; Flask reads it for the live UI.

USAGE
─────
# Normal run (Arduino on /dev/ttyUSB0):
  python main.py recipes/pasta.yaml

# Specify serial port (Pi often uses ttyACM0):
  python main.py recipes/pasta.yaml --port /dev/ttyACM0

# Mock mode — no Arduino needed, motor commands printed:
  python main.py recipes/pasta.yaml --mock

# Mock camera mode — synthetic frames (useful in dev containers):
  python main.py recipes/pasta.yaml --mock-cam

# Headless — operator confirms only via web UI, no keyboard prompts:
  python main.py recipes/pasta.yaml --headless

# Custom camera index (e.g. USB cam is /dev/video1):
  python main.py recipes/pasta.yaml --cam 1

# Custom UI port:
  python main.py recipes/pasta.yaml --ui-port 8080

# Full example for a Pi with no Arduino yet:
  python main.py recipes/pasta.yaml --mock --headless
"""

import argparse
import time

from ui_server   import start_server
from orchestrator import Orchestrator


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="COOKi OS — automated cooking robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("recipe",      help="Path to recipe YAML file")
    ap.add_argument("--port",      default="/dev/ttyUSB0",
                    help="Arduino serial port (default: /dev/ttyUSB0)")
    ap.add_argument("--mock",      action="store_true",
                    help="No Arduino — motor commands printed only")
    ap.add_argument("--mock-cam",  action="store_true",
            help="No webcam needed — use synthetic camera frames")
    ap.add_argument("--headless",  action="store_true",
                    help="Confirm ingredients via UI only (no keyboard)")
    ap.add_argument("--cam",       type=int, default=0,
                    help="Camera index (default: 0)")
    ap.add_argument("--ui-port",   type=int, default=5000,
                    help="Web dashboard port (default: 5000)")
    args = ap.parse_args()

    # Start Flask dashboard in a daemon thread
    start_server(port=args.ui_port)
    time.sleep(0.6)         # let Flask bind before orchestrator starts

    # Run the recipe
    cook = Orchestrator(
        recipe_path=args.recipe,
        motor_port=args.port,
        mock=args.mock,
        headless=args.headless,
        cam_index=args.cam,
        mock_camera=args.mock_cam,
    )
    cook.run()
