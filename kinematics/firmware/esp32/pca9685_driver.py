"""
PCA9685 16-channel PWM driver for MicroPython (ESP32).

Usage:
    from machine import I2C, Pin
    from pca9685_driver import PCA9685

    i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400_000)
    pca = PCA9685(i2c, address=0x40)
    pca.set_pwm_freq(50)                  # 50 Hz for servos
    pca.set_pulse_us(channel=0, pulse_us=1500)  # centre a servo
"""

import time

# ── Register map ──────────────────────────────────────────────────────────────
_MODE1      = 0x00
_MODE2      = 0x01
_PRESCALE   = 0xFE
_LED0_ON_L  = 0x06   # base of per-channel on/off registers (4 bytes each)

_SLEEP      = 0x10
_RESTART    = 0x80
_AI         = 0x20   # auto-increment flag
_ALLCALL    = 0x01
_OUTDRV     = 0x04   # totem-pole output (required for most servo drivers)

_PWM_FREQ   = 50     # default servo frequency (Hz)


class PCA9685:
    """
    Driver for the PCA9685 16-channel 12-bit PWM controller.

    All timing is expressed in microseconds at the board level so the
    caller never needs to know the resolution or prescaler value.
    """

    def __init__(self, i2c, address: int = 0x40, freq_hz: int = _PWM_FREQ):
        self._i2c   = i2c
        self._addr  = address
        self._freq  = freq_hz
        self._reset()
        self.set_pwm_freq(freq_hz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pwm_freq(self, freq_hz: int):
        """Set the PWM frequency for all channels (Hz). Typical: 50 Hz."""
        self._freq = freq_hz
        prescale   = round(25_000_000.0 / (4096 * freq_hz)) - 1
        old_mode   = self._read(_MODE1)
        self._write(_MODE1, (old_mode & 0x7F) | _SLEEP)   # enter sleep
        self._write(_PRESCALE, prescale)
        self._write(_MODE1, old_mode)
        time.sleep_ms(5)
        self._write(_MODE1, old_mode | _RESTART)

    def set_pulse_us(self, channel: int, pulse_us: float):
        """
        Set the pulse width (µs) on one channel.

        At 50 Hz (20 ms period):
            500  µs → ~102 counts  (≈ 0° for most servos)
            1500 µs → ~307 counts  (≈ 90° / centre)
            2500 µs → ~512 counts  (≈ 180°)
        """
        period_us = 1_000_000.0 / self._freq
        off = round(pulse_us / period_us * 4096)
        off = max(0, min(4095, off))
        self._set_channel_raw(channel, 0, off)

    def set_all_centre(self, centre_us: float = 1500.0):
        """Send all 16 channels to the given centre pulse (µs)."""
        for ch in range(16):
            self.set_pulse_us(ch, centre_us)

    def disable_channel(self, channel: int):
        """Drive channel fully off (no pulse — servo goes limp)."""
        self._set_channel_raw(channel, 0, 0)

    def disable_all(self):
        for ch in range(16):
            self.disable_channel(ch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self):
        self._write(_MODE1, _ALLCALL | _AI)
        self._write(_MODE2, _OUTDRV)
        time.sleep_ms(5)
        mode1 = self._read(_MODE1) & ~_SLEEP
        self._write(_MODE1, mode1)
        time.sleep_ms(5)

    def _set_channel_raw(self, channel: int, on: int, off: int):
        reg = _LED0_ON_L + 4 * channel
        self._i2c.writeto_mem(
            self._addr, reg,
            bytes([on & 0xFF, on >> 8, off & 0xFF, off >> 8]),
        )

    def _write(self, reg: int, val: int):
        self._i2c.writeto_mem(self._addr, reg, bytes([val]))

    def _read(self, reg: int) -> int:
        return self._i2c.readfrom_mem(self._addr, reg, 1)[0]
