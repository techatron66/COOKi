"""
motor.py — COOKi OS  |  Arduino serial bridge

Protocol (newline-terminated):   "speed,direction\n"
  speed:     0–255 (PWM)
  direction: 0 (CCW) or 1 (CW)

Arduino responds:  "OK spd=200 dir=1"

Run with --mock to skip serial (motor commands printed to console).
"""

import serial
import time
import threading


class MotorController:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 9600,
                 mock: bool = False):
        self.mock  = mock
        self._lock = threading.Lock()

        if mock:
            print("[MOTOR] Mock mode — no Arduino needed")
            self._speed = 0
            self._direction = 0
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)          # wait for Arduino reset after DTR pulse
            print(f"[MOTOR] Connected → {port} @ {baud} baud")
        except serial.SerialException as e:
            raise RuntimeError(
                f"Cannot open {port}. Check connection or use --mock.\n{e}"
            )

    def send(self, speed: int, direction: int):
        """Send motor command.  speed 0-255, direction 0 or 1."""
        speed     = max(0, min(255, int(speed)))
        direction = 1 if direction else 0

        if self.mock:
            with self._lock:
                self._speed     = speed
                self._direction = direction
            print(f"[MOTOR MOCK] spd={speed:3d}  dir={direction}")
            return

        cmd = f"{speed},{direction}\n"
        with self._lock:
            self.ser.write(cmd.encode())

    def stop(self):
        self.send(0, 0)

    def close(self):
        self.stop()
        if not self.mock and hasattr(self, "ser"):
            self.ser.close()
