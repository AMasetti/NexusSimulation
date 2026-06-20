"""
Interactive servo calibration helper.

Walks through each servo channel, lets you manually set a pulse width,
and records the centre/min/max values. Results are printed as a
servo_config.py snippet ready to paste in.

Run:
    python calibrate_servos.py --port /dev/ttyUSB0

Controls (per channel):
    <number>   →  send that pulse width in µs  (e.g. 1500)
    +          →  increase by step (default 10 µs)
    -          →  decrease by step
    s <n>      →  change step size to n µs
    c          →  record this pulse as CENTRE (joint angle = 0)
    mn         →  record this pulse as MIN (most negative joint angle)
    mx         →  record this pulse as MAX (most positive joint angle)
    n          →  next channel
    b          →  back to previous channel
    q          →  quit and print results
"""

import argparse
import json
import sys
import time
import serial

# ── Joint metadata (for display only) ────────────────────────────────────────
JOINT_LABELS = [
    "FL shoulder", "FL hip", "FL knee",
    "FR shoulder", "FR hip", "FR knee",
    "RL shoulder", "RL hip", "RL knee",
    "RR shoulder", "RR hip", "RR knee",
    "AUX 12",      "AUX 13",
]

DEFAULT_PORT     = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
N_CHANNELS       = 14


def send(ser, channel, pulse_us):
    """Send a single servo pulse via the ESP32 firmware."""
    msg = json.dumps({"cmd": "angles", "data": _angles_with_pulse(channel, pulse_us)}) + "\n"
    ser.write(msg.encode())
    ser.readline()   # consume response


def _angles_with_pulse(channel, pulse_us):
    """Build a 12-element angles list with only one channel active."""
    # We abuse the angles command by sending a placeholder; the firmware
    # maps index → channel. For calibration we talk directly to the
    # PCA9685 via a dedicated 'raw_pulse' command instead.
    # This function is kept for documentation; calibrate_servos uses its
    # own raw_pulse path below.
    return [0.0] * 12


def send_raw(ser, channel: int, pulse_us: float):
    """
    Send a raw pulse (µs) to one PCA9685 channel.
    Requires the ESP32 firmware to support {"cmd": "raw_pulse", "ch": N, "us": F}.
    If not supported, falls back to printing a manual instruction.
    """
    msg = json.dumps({"cmd": "raw_pulse", "ch": channel, "us": pulse_us}) + "\n"
    ser.write(msg.encode())
    raw = ser.readline()
    try:
        resp = json.loads(raw.decode().strip())
        if not resp.get("ok"):
            # Firmware may not have this command yet — print manual instructions
            print(f"    [fallback] set ch {channel} to {pulse_us:.0f} µs manually")
    except Exception:
        pass


def calibrate(port, baudrate):
    ser = serial.Serial(port, baudrate, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()

    results = {}   # channel → {centre, min_us, max_us}
    for ch in range(N_CHANNELS):
        results[ch] = {"centre_us": 1500, "min_us": 500, "max_us": 2500, "direction": 1}

    pulse   = 1500
    step    = 10
    channel = 0

    print(f"\nCalibrating {N_CHANNELS} channels on {port}")
    print("Commands: <µs> | + | - | s<step> | c | mn | mx | n | b | q\n")

    while channel < N_CHANNELS:
        label = JOINT_LABELS[channel] if channel < len(JOINT_LABELS) else f"ch {channel}"
        print(f"── Channel {channel}  ({label})  current={pulse} µs ──")
        send_raw(ser, channel, pulse)

        while True:
            try:
                raw = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                channel = N_CHANNELS   # exit
                break

            if not raw:
                continue

            if raw == "q":
                channel = N_CHANNELS
                break
            elif raw == "n":
                channel += 1
                pulse    = 1500
                break
            elif raw == "b":
                channel  = max(0, channel - 1)
                pulse    = 1500
                break
            elif raw == "+":
                pulse += step
            elif raw == "-":
                pulse -= step
            elif raw.startswith("s") and raw[1:].isdigit():
                step = int(raw[1:])
                print(f"  step = {step} µs")
                continue
            elif raw == "c":
                results[channel]["centre_us"] = pulse
                print(f"  ✓ centre = {pulse} µs")
                continue
            elif raw == "mn":
                results[channel]["min_us"] = pulse
                print(f"  ✓ min    = {pulse} µs")
                continue
            elif raw == "mx":
                results[channel]["max_us"] = pulse
                print(f"  ✓ max    = {pulse} µs")
                continue
            elif raw.lstrip("-").isdigit():
                pulse = int(raw)
            else:
                print("  unknown — try: 1500 | + | - | s10 | c | mn | mx | n | b | q")
                continue

            pulse = max(400, min(2600, pulse))
            send_raw(ser, channel, pulse)
            print(f"  pulse = {pulse} µs", end="\r")

    ser.close()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n\n── Calibration results — paste into servo_config.py ────────────\n")
    import math
    scale_default = 2000.0 / math.pi

    for ch in range(N_CHANNELS):
        r     = results[ch]
        label = JOINT_LABELS[ch] if ch < len(JOINT_LABELS) else f"ch{ch}"
        # Infer scale from min/max if both were set
        if r["min_us"] != 500 or r["max_us"] != 2500:
            half_range_us  = (r["max_us"] - r["min_us"]) / 2.0
            scale          = half_range_us / (math.pi / 2)
        else:
            scale = scale_default
        print(
            f"    _servo(channel={ch:2d}, direction={r['direction']:+d}, "
            f"center_us={r['centre_us']}, "
            f"scale_us_per_rad={scale:.1f}, "
            f"min_us={r['min_us']}, max_us={r['max_us']}),  # {label}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servo calibration helper")
    parser.add_argument("--port",     default=DEFAULT_PORT)
    parser.add_argument("--baudrate", default=DEFAULT_BAUDRATE, type=int)
    args = parser.parse_args()
    calibrate(args.port, args.baudrate)
