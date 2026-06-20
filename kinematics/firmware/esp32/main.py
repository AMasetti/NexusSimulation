"""
ESP32 firmware — SpotMicro servo controller.

Receives joint-angle commands over serial (USB) or WiFi (optional),
converts them to MG995 PWM signals via PCA9685, and applies them.

── Serial protocol (newline-delimited JSON) ──────────────────────────
  Host → ESP32:
    {"cmd": "angles",  "data": [a0, a1, ..., a11]}   12 floats (rad)
    {"cmd": "pose",    "pos":  [x,y,z], "euler": [r,p,y]}
    {"cmd": "halt"}           safe-stop: hold current position
    {"cmd": "centre"}         move all servos to 1500 µs
    {"cmd": "ping"}           connectivity check

  ESP32 → Host:
    {"ok": true}
    {"ok": false, "error": "<message>"}

── Flash to ESP32 ────────────────────────────────────────────────────
  Copy pca9685_driver.py and servo_config.py alongside this file,
  then upload with ampy, mpremote, or Thonny:
      mpremote cp pca9685_driver.py :
      mpremote cp servo_config.py   :
      mpremote cp main.py           :main.py   # runs on boot
"""

import json
import sys
import math
import time

from machine import I2C, Pin
from pca9685_driver import PCA9685
from servo_config import SERVO_MAP, angle_to_pulse_us

# ── Hardware configuration ────────────────────────────────────────────────────
I2C_ID  = 0
SCL_PIN = 22
SDA_PIN = 21
I2C_FREQ = 400_000
PCA_ADDR = 0x40

# ── WiFi (optional — leave SSID empty to disable) ────────────────────────────
WIFI_SSID = ""
WIFI_PASS = ""

# ── Init ──────────────────────────────────────────────────────────────────────
i2c = I2C(I2C_ID, scl=Pin(SCL_PIN), sda=Pin(SDA_PIN), freq=I2C_FREQ)

devices = i2c.scan()
if PCA_ADDR not in devices:
    print(json.dumps({"ok": False, "error": f"PCA9685 not found at 0x{PCA_ADDR:02X}. Found: {[hex(d) for d in devices]}"}))
    sys.exit()

pca = PCA9685(i2c, address=PCA_ADDR, freq_hz=50)
pca.set_all_centre()   # start with all servos centred
print(json.dumps({"ok": True, "msg": "PCA9685 ready"}))

# ── WiFi setup ────────────────────────────────────────────────────────────────
if WIFI_SSID:
    import network
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASS)
    t = 0
    while not wlan.isconnected() and t < 10:
        time.sleep(1)
        t += 1
    if wlan.isconnected():
        print(json.dumps({"ok": True, "msg": f"WiFi connected: {wlan.ifconfig()[0]}"}))
    else:
        print(json.dumps({"ok": False, "error": "WiFi connect timeout"}))


# ── Command handlers ──────────────────────────────────────────────────────────

def apply_angles(angles):
    """Apply 12 joint angles (rad) to the corresponding servo channels."""
    if len(angles) != len(SERVO_MAP):
        raise ValueError(f"Expected {len(SERVO_MAP)} angles, got {len(angles)}")
    for cfg, angle in zip(SERVO_MAP, angles):
        pulse = angle_to_pulse_us(angle, cfg)
        pca.set_pulse_us(cfg["channel"], pulse)


def handle(msg: dict):
    cmd = msg.get("cmd", "")

    if cmd == "ping":
        return {"ok": True, "msg": "pong"}

    if cmd == "centre":
        pca.set_all_centre()
        return {"ok": True}

    if cmd == "halt":
        # Hold current position — do nothing (servos stay where they are)
        return {"ok": True, "msg": "holding"}

    if cmd == "angles":
        angles = msg.get("data", [])
        apply_angles(angles)
        return {"ok": True}

    if cmd == "raw_pulse":
        ch = msg.get("ch")
        us = msg.get("us")
        if ch is None or us is None:
            return {"ok": False, "error": "raw_pulse requires 'ch' and 'us'"}
        pca.set_pulse_us(int(ch), float(us))
        return {"ok": True}

    if cmd == "pose":
        # Minimal on-device IK not implemented — host should send angles directly.
        # This slot is reserved for future on-device computation.
        return {"ok": False, "error": "on-device IK not supported; send 'angles' instead"}

    return {"ok": False, "error": f"unknown command: {cmd!r}"}


# ── Main serial loop ──────────────────────────────────────────────────────────
buf = b""
while True:
    chunk = sys.stdin.buffer.read(64)
    if chunk:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg  = json.loads(line)
                resp = handle(msg)
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(resp) + "\n")
