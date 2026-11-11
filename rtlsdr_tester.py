#!/usr/bin/env python3
"""
rtlsdr_tester.py - GUI test bench for NooElec NESDR / RTL2832U dongles.

What it does
  * Detects the dongle and shows its USB strings, tuner chip and gain table
  * Live spectrum + waterfall (click on either to tune there)
  * Controls: frequency, presets, sample rate, gain/AGC, PPM, FFT size, bias-tee
  * Live stream health: real throughput, dropped blocks, ADC level, clipping,
    DC offset, I/Q balance, strongest peak
  * Wideband-FM and narrowband-AM audio receivers, to check reception by listening
  * Automated self-test:
        - USB integrity using the RTL2832 built-in counter test mode (like rtl_test)
        - sustained sample-rate sweep
        - tuner tuning sweep across the tuner's range
        - gain response (LNA/mixer/VGA chain works)
        - raw ADC sample sanity (stuck bits, DC, I/Q imbalance, clipping)

Requirements
  pip install "pyrtlsdr[lib]" numpy scipy matplotlib sounddevice
  On Windows, use Zadig to bind WinUSB to "Bulk-In, Interface (Interface 0)".
  Close any other SDR software (SDR#, SDR++, ...) first: only one program can
  open the dongle at a time.
"""

import ctypes
import math
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from scipy import signal  # noqa: E402

try:
    from rtlsdr import RtlSdr
    from rtlsdr.librtlsdr import librtlsdr as _lib

    RTLSDR_IMPORT_ERROR = None
except Exception as exc:  # package missing or librtlsdr DLL not found
    RtlSdr = None
    _lib = None
    RTLSDR_IMPORT_ERROR = exc

try:
    import sounddevice as sd
except Exception:  # audio is optional
    sd = None


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TUNER_NAMES = {
    0: "Unknown",
    1: "Elonics E4000",
    2: "Fitipower FC0012",
    3: "Fitipower FC0013",
    4: "FCI FC2580",
    5: "Rafael Micro R820T/R820T2",
    6: "Rafael Micro R828D",
}
# Approximate usable tuning range per tuner, in MHz
TUNER_RANGES = {
    1: (52, 2200),
    2: (22, 948),
    3: (22, 1100),
    4: (146, 924),
    5: (24, 1766),
    6: (24, 1766),
}
SAMPLE_RATES = [0.96e6, 1.024e6, 1.2e6, 1.44e6, 1.92e6, 2.048e6, 2.4e6, 2.88e6, 3.2e6]
FFT_SIZES = [512, 1024, 2048, 4096]
PRESETS = [
    ("FM broadcast 100.0 MHz", 100.0),
    ("FM broadcast 88.0 MHz", 88.0),
    ("Airband 125.0 MHz", 125.0),
    ("DAB block 12C 227.36 MHz", 227.36),
    ("ISM 433.92 MHz", 433.92),
    ("PMR446 446.1 MHz", 446.1),
    ("ISM 868.3 MHz", 868.3),
    ("GSM900 downlink 945 MHz", 945.0),
    ("ADS-B 1090 MHz", 1090.0),
]
IF_RATE = 240_000  # WFM intermediate rate
AUDIO_RATE = 48_000
WF_ROWS = 200
WF_MAX_COLS = 1024


def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def list_devices():
    """Enumerate RTL-SDR devices with their USB strings."""
    devs = []
    for i in range(_lib.rtlsdr_get_device_count()):
        name = _lib.rtlsdr_get_device_name(i)
        name = name.decode(errors="replace") if name else "?"
        bufs = [(ctypes.c_ubyte * 256)() for _ in range(3)]
        r = _lib.rtlsdr_get_device_usb_strings(i, *bufs)
        if r == 0:
            mfr, prod, ser = (bytes(b).split(b"\0", 1)[0].decode(errors="replace") for b in bufs)
        else:
            mfr = prod = ser = "?"
        devs.append(dict(index=i, name=name, manufacturer=mfr, product=prod, serial=ser))
    return devs


def set_ppm(sdr, ppm):
    """librtlsdr rejects setting the same ppm twice (and pyrtlsdr then closes the device)."""
    ppm = int(ppm)
    if sdr.get_freq_correction() != ppm:
        sdr.freq_correction = ppm


def apply_gain(sdr, gain):
    if gain == "auto":
        sdr.gain = "auto"
        sdr.set_agc_mode(True)
    else:
        sdr.set_agc_mode(False)
        sdr.gain = float(gain)


# --------------------------------------------------------------------------
# Extended librtlsdr API
# --------------------------------------------------------------------------
# pyrtlsdrlib bundles the "librtlsdr/librtlsdr" fork, which exports far more
# controls than pyrtlsdr wraps (per-stage gains, IF mode, sideband, tuner
# registers, EEPROM, ...). Declare their C prototypes so ctypes passes the
# arguments correctly. Functions missing from another librtlsdr build are skipped.
def _bind_ext():
    if _lib is None:
        return set()
    from rtlsdr.librtlsdr import p_rtlsdr_dev as D

    c = ctypes
    protos = {
        "rtlsdr_get_ver_id": (c.c_char_p, []),
        "rtlsdr_get_version": (c.c_uint32, []),
        "rtlsdr_get_opt_help": (c.c_char_p, [c.c_int]),
        "rtlsdr_set_opt_string": (c.c_int, [D, c.c_char_p, c.c_int]),
        "rtlsdr_is_tuner_PLL_locked": (c.c_int, [D]),
        "rtlsdr_get_direct_sampling": (c.c_int, [D]),
        "rtlsdr_set_ds_mode": (c.c_int, [D, c.c_int, c.c_uint32]),
        "rtlsdr_set_offset_tuning": (c.c_int, [D, c.c_int]),
        "rtlsdr_get_offset_tuning": (c.c_int, [D]),
        "rtlsdr_set_tuner_gain_ext": (c.c_int, [D, c.c_int, c.c_int, c.c_int]),
        "rtlsdr_set_tuner_if_gain": (c.c_int, [D, c.c_int, c.c_int]),
        "rtlsdr_set_tuner_if_mode": (c.c_int, [D, c.c_int]),
        "rtlsdr_set_tuner_sideband": (c.c_int, [D, c.c_int]),
        "rtlsdr_set_tuner_band_center": (c.c_int, [D, c.c_int32]),
        "rtlsdr_set_harmonic_rx": (c.c_int, [D, c.c_int]),
        "rtlsdr_get_tuner_i2c_register": (c.c_int, [D, c.POINTER(c.c_ubyte), c.c_int]),
        "rtlsdr_set_tuner_i2c_register": (c.c_int, [D, c.c_uint, c.c_uint, c.c_uint]),
        "rtlsdr_set_tuner_i2c_override": (c.c_int, [D, c.c_uint, c.c_uint, c.c_uint]),
        "rtlsdr_read_eeprom": (c.c_int, [D, c.POINTER(c.c_ubyte), c.c_uint8, c.c_uint16]),
        "rtlsdr_set_and_get_tuner_bandwidth": (c.c_int, [D, c.c_uint32, c.POINTER(c.c_uint32), c.c_int]),
    }
    found = set()
    for name, (res, args) in protos.items():
        f = getattr(_lib, name, None)
        if f is not None:
            f.restype, f.argtypes = res, args
            found.add(name)
    return found


EXT = _bind_ext()
R82XX_TUNERS = (5, 6)
ZERO_IF_TUNERS = (1, 2, 3, 4)  # offset tuning only makes sense on these
DS_MODES = [
    ("Off (I/Q through the tuner)", 0),
    ("I branch", 1),
    ("Q branch", 2),
    ("I branch below threshold", 3),
    ("Q branch below threshold", 4),
]
BANDWIDTHS_KHZ = [290, 375, 420, 470, 600, 860, 950, 1100, 1300, 1500, 1600, 1750, 1950,
                  2200, 3000, 5000, 6000, 7000, 8000]
GPIO_NOTES = {
    0: "bias-tee power on most NooElec / v3 dongles",
    4: "tuner reset: driving it disables the tuner",
    6: "FC0012 V/UHF band switch",
}


def chk(name, r):
    """Raise when a librtlsdr call returns a negative error code."""
    if r < 0:
        raise RuntimeError(f"{name}() returned error {r}")
    return r


def xcall(name, *args):
    """Call an extended librtlsdr function, failing clearly when it is missing."""
    f = getattr(_lib, name, None) if name in EXT else None
    if f is None:
        raise RuntimeError(f"{name}() is not available in this librtlsdr build")
    return chk(name, f(*args))


def lib_version():
    if "rtlsdr_get_ver_id" not in EXT:
        return "unknown"
    ver = _lib.rtlsdr_get_version() if "rtlsdr_get_version" in EXT else 0
    vid = (_lib.rtlsdr_get_ver_id() or b"?").decode(errors="replace")
    return f"{ver >> 16}.{ver & 0xFFFF}  {vid}"


def read_radio_state(sdr):
    """Read back everything the driver can report. Values that cannot be read
    are stored as the exception that occurred."""
    d = sdr.dev_p
    st = {}

    def grab(key, fn):
        try:
            st[key] = fn()
        except Exception as exc:
            st[key] = exc

    def xtal():
        a, b = ctypes.c_uint(), ctypes.c_uint()
        chk("rtlsdr_get_xtal_freq", _lib.rtlsdr_get_xtal_freq(d, ctypes.byref(a), ctypes.byref(b)))
        return a.value, b.value

    def gpio():
        v = ctypes.c_int()
        chk("rtlsdr_get_gpio_byte", _lib.rtlsdr_get_gpio_byte(d, ctypes.byref(v)))
        return v.value

    def i2c():
        buf = (ctypes.c_ubyte * 32)()
        xcall("rtlsdr_get_tuner_i2c_register", d, buf, 32)
        return bytes(buf)

    def pll():
        if "rtlsdr_is_tuner_PLL_locked" not in EXT:
            raise RuntimeError("not available")
        return _lib.rtlsdr_is_tuner_PLL_locked(d)

    grab("tuner", lambda: _lib.rtlsdr_get_tuner_type(d))
    grab("center_freq", lambda: _lib.rtlsdr_get_center_freq(d))
    grab("sample_rate", lambda: _lib.rtlsdr_get_sample_rate(d))
    grab("ppm", lambda: _lib.rtlsdr_get_freq_correction(d))
    grab("tuner_gain", lambda: _lib.rtlsdr_get_tuner_gain(d) / 10.0)
    grab("xtal", xtal)
    grab("direct_sampling", lambda: xcall("rtlsdr_get_direct_sampling", d))
    grab("offset_tuning", lambda: xcall("rtlsdr_get_offset_tuning", d))
    grab("pll", pll)
    grab("gpio", gpio)
    grab("i2c", i2c)
    return st


def read_eeprom(sdr, size=256):
    buf = (ctypes.c_ubyte * size)()
    xcall("rtlsdr_read_eeprom", sdr.dev_p, buf, 0, size)
    return bytes(buf)


def decode_eeprom(b):
    """Decode the RTL2832U EEPROM header (same layout rtl_eeprom uses)."""
    if len(b) < 9 or b[0] != 0x28 or b[1] != 0x32:
        return ["No valid RTL2832U EEPROM header (expected 28 32)"]
    lines = [
        f"VID:PID  {b[2] | b[3] << 8:04X}:{b[4] | b[5] << 8:04X}",
        f"Serial number enabled: {'yes' if b[6] == 0xA5 else 'no'}",
        f"Remote wakeup: {'on' if b[7] & 1 else 'off'}   IR receiver: {'on' if b[7] & 2 else 'off'}",
    ]
    pos = 9
    for label in ("Manufacturer", "Product", "Serial"):
        if pos + 2 > len(b):
            break
        n = b[pos]
        if n < 2 or b[pos + 1] != 0x03 or pos + n > len(b):
            break
        lines.append(f"{label}: {b[pos + 2 : pos + n].decode('utf-16-le', errors='replace')}")
        pos += n
    return lines


def hexdump(b, width=16, ascii_col=True):
    out = []
    for i in range(0, len(b), width):
        chunk = b[i : i + width]
        line = f"{i:02X}: " + " ".join(f"{x:02X}" for x in chunk).ljust(width * 3 - 1)
        if ascii_col:
            line += "  " + "".join(chr(x) if 32 <= x < 127 else "." for x in chunk)
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# DSP blocks
# --------------------------------------------------------------------------
class FirDecimator:
    """Stateful FIR low-pass + integer decimation (no clicks at block edges)."""

    def __init__(self, taps, dec, dtype):
        self.taps = np.asarray(taps)
        self.dec = int(dec)
        self.hist = np.zeros(len(taps) - 1, dtype=dtype)
        self.phase = 0

    def process(self, x):
        xe = np.concatenate((self.hist, x))
        y = signal.oaconvolve(xe, self.taps, mode="valid")  # len(y) == len(x)
        out = y[self.phase :: self.dec]
        self.phase = self.phase + len(out) * self.dec - len(y)
        self.hist = xe[-(len(self.taps) - 1) :]
        return out


class WfmDemod:
    """Mono wideband-FM demodulator: fs -> 240 kHz -> discriminator -> de-emphasis -> 48 kHz."""

    def __init__(self, fs, deemph_us=50.0):
        if int(fs) % IF_RATE:
            raise ValueError("sample rate must be a multiple of 240 kHz for FM audio")
        d1 = int(fs) // IF_RATE
        self.stage1 = FirDecimator(signal.firwin(16 * d1 + 1, 100e3, fs=fs), d1, np.complex128)
        self.stage2 = FirDecimator(signal.firwin(101, 15e3, fs=IF_RATE), IF_RATE // AUDIO_RATE, np.float64)
        a = math.exp(-1.0 / (IF_RATE * deemph_us * 1e-6))
        self.de_b, self.de_a = [1 - a], [1, -a]
        self.de_zi = np.zeros(1)
        self.prev = 0j
        self.max_dev = 2 * math.pi * 75e3 / IF_RATE  # radians/sample at 75 kHz deviation

    def process(self, iq):
        x = self.stage1.process(iq)
        if len(x) == 0:
            return np.zeros(0, np.float32)
        xx = np.concatenate(([self.prev], x))
        self.prev = x[-1]
        d = np.angle(xx[1:] * np.conj(xx[:-1])) / self.max_dev
        d, self.de_zi = signal.lfilter(self.de_b, self.de_a, d, zi=self.de_zi)
        return self.stage2.process(d).astype(np.float32)


class AirbandAmDemod:
    """AM voice demodulator for the VHF airband, producing 48 kHz mono audio.

    The RTL is tuned to the desired AM carrier.  A narrow complex low-pass
    selects one 8.33/25 kHz airband channel, then an envelope detector and a
    voice-band filter recover speech.  Keeping the filter and decimator state
    across USB blocks avoids clicks between reads.
    """

    def __init__(self, fs):
        dec = round(fs / AUDIO_RATE)
        if dec < 1 or not math.isclose(fs, dec * AUDIO_RATE, rel_tol=0, abs_tol=1):
            raise ValueError("AM audio needs a sample rate that is a multiple of 48 kHz")
        # 10 kHz is deliberately wide enough for a 25 kHz AM channel while
        # still rejecting adjacent airband channels before envelope detection.
        self.stage = FirDecimator(signal.firwin(16 * dec + 1, 10e3, fs=fs), dec, np.complex128)
        self.voice_sos = signal.butter(4, (250, 4_500), btype="bandpass", fs=AUDIO_RATE, output="sos")
        self.voice_zi = signal.sosfilt_zi(self.voice_sos) * 0.0
        self.carrier = None

    def process(self, iq):
        x = self.stage.process(iq)
        if len(x) == 0:
            return np.zeros(0, np.float32)
        envelope = np.abs(x)
        # Slowly follow the carrier level so manual gain changes do not make
        # speech abruptly too quiet or clip.  Normalising before the voice
        # filter also removes most AM receiver gain variation.
        level = float(np.mean(envelope))
        self.carrier = level if self.carrier is None else 0.98 * self.carrier + 0.02 * level
        audio = envelope / max(self.carrier, 1e-4) - 1.0
        audio, self.voice_zi = signal.sosfilt(self.voice_sos, audio, zi=self.voice_zi)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


class AudioSink:
    """sounddevice output fed from a jitter buffer."""

    def __init__(self, rate=AUDIO_RATE):
        self.rate = rate
        self.chunks = deque()
        self.queued = 0
        self.lock = threading.Lock()
        self.starved = True
        self.underruns = 0
        self.volume = 0.5
        self.stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", blocksize=1024,
            latency="high", callback=self._callback,
        )
        self.stream.start()

    def push(self, a):
        with self.lock:
            self.chunks.append(a)
            self.queued += len(a)
            # keep latency bounded (RTL clock vs sound card clock drift)
            while self.queued > self.rate * 0.6 and self.chunks:
                self.queued -= len(self.chunks.popleft())

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        out[:] = 0
        with self.lock:
            if self.starved:
                if self.queued < self.rate * 0.15:
                    return
                self.starved = False
            n = 0
            while n < frames and self.chunks:
                c = self.chunks[0]
                take = min(frames - n, len(c))
                out[n : n + take] = c[:take]
                n += take
                self.queued -= take
                if take == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[take:]
            if n < frames:
                self.starved = True
                self.underruns += 1
        np.clip(out * self.volume, -1, 1, out=out)

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Streaming engine
# --------------------------------------------------------------------------
class SdrStream:
    """Owns the device while streaming: USB reader thread + DSP thread.

    The reader uses synchronous bulk reads in a tight loop (like `rtl_sdr -S`):
    the async API of the librtlsdr build bundled with pyrtlsdrlib crashes inside
    libusb on Windows. Control commands (tune, gain, ...) are queued and executed
    by the reader thread between reads, so the device is only touched by one thread.
    """

    def __init__(self, index, fs, fc, gain, ppm, nfft, bias_tee=False, dithering=True):
        self.fs = fs
        self.fc = fc
        self.nfft = nfft
        self.cmd_q = queue.Queue()
        # PLL dithering can only be changed before the first tune, i.e. at open
        self.sdr = RtlSdr(device_index=index, dithering_enabled=dithering)
        try:
            self.sdr.sample_rate = fs
            self.sdr.center_freq = fc
            set_ppm(self.sdr, ppm)
            apply_gain(self.sdr, gain)
            if bias_tee:
                self.sdr.set_bias_tee(True)
        except Exception:
            self.sdr.close()
            raise
        self.block = 16384 * max(1, round(fs * 0.065 / 16384))  # ~65 ms per USB read
        self.q = queue.Queue(maxsize=64)
        self.stopping = False
        self.error = None
        self.total_samples = 0
        self.dropped_blocks = 0
        self.psd_lock = threading.Lock()
        self.psd = None
        self.psd_seq = 0
        self.burst_envelope = None
        self.envelope_q = queue.Queue(maxsize=256)
        self.envelope_bin_s = 0.0001  # display-only power averaging interval
        self.stats = {}
        self.demod = None
        self.audio = None
        self.audio_mode = None

    # -- lifecycle --
    def start(self):
        self.t0 = time.perf_counter()
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.dsp = threading.Thread(target=self._dsp, daemon=True)
        self.dsp.start()
        self.reader.start()

    def stop(self):
        self.stopping = True
        self.reader.join(3)
        self.q.put(None)
        self.dsp.join(3)
        self.set_audio(False)
        if self.reader.is_alive():
            return False  # never close a handle librtlsdr is still using
        if not self.sdr.device_opened:
            return True
        _lib.rtlsdr_reset_buffer(self.sdr.dev_p)  # see _reader: makes register writes reliable
        try:
            self.sdr.set_bias_tee(False)
        except Exception:
            pass
        self.sdr.close()
        return True

    # -- control (called from GUI thread; executed by the reader thread) --
    def control(self, what, fn, *args):
        self.cmd_q.put((what, fn, args))

    def tune(self, fc):
        self.sdr.center_freq = fc
        self.fc = fc

    def set_audio(self, enabled, volume=0.5, mode="wfm"):
        """Enable one audio receiver at a time, replacing any existing one."""
        self.demod = None
        self.audio_mode = None
        old_audio, self.audio = self.audio, None
        if old_audio:
            old_audio.close()
        if enabled:
            demodulators = {"wfm": WfmDemod, "am": AirbandAmDemod}
            try:
                self.demod = demodulators[mode](self.fs)
            except KeyError:
                raise ValueError(f"Unknown audio mode: {mode}") from None
            self.audio = AudioSink()
            self.audio.volume = volume
            self.audio_mode = mode

    # -- threads --
    def _reader(self):
        nbytes = 2 * self.block
        try:
            _lib.rtlsdr_reset_buffer(self.sdr.dev_p)
            while not self.stopping:
                if not self.cmd_q.empty():
                    # Register writes issued while the bulk endpoint is streaming
                    # fail with a USB pipe error (and pyrtlsdr then closes the
                    # device). Stopping/resetting the endpoint first avoids it.
                    for _ in range(3):
                        if _lib.rtlsdr_reset_buffer(self.sdr.dev_p) == 0:
                            break
                # coalesce: only the latest command of each kind matters
                # (a retune takes ~75 ms, so slider drags would pile up).
                # Re-inserting moves a repeated command to the end, so commands
                # still run in the order they were last issued.
                pending = {}
                while not self.cmd_q.empty():
                    what, fn, args = self.cmd_q.get_nowait()
                    pending.pop(what, None)
                    pending[what] = (fn, args)
                for what, (fn, args) in pending.items():
                    try:
                        fn(*args)
                    except Exception as exc:
                        self.error = f"{what} failed: {exc}"
                        if not self.sdr.device_opened:  # pyrtlsdr closes on some errors
                            return
                buf = self.sdr.read_bytes(nbytes)
                self.total_samples += self.block
                try:
                    self.q.put_nowait(buf)
                except queue.Full:
                    self.dropped_blocks += 1
        except Exception as exc:
            if not self.stopping:
                self.error = f"USB read failed: {exc}"

    def _dsp(self):
        while True:
            buf = self.q.get()
            if buf is None:
                return
            try:
                raw = np.ctypeslib.as_array(buf)
                iq = (raw.astype(np.float32) * (1 / 127.5) - 1.0).view(np.complex64)
                self._process(iq)
            except Exception as exc:  # keep streaming, report it
                self.error = f"DSP error: {exc!r}"

    def _process(self, iq):
        nfft = self.nfft
        # ---- spectrum (average up to 32 FFT frames of this block) ----
        nframes = min(32, len(iq) // nfft)
        psd = None
        if nframes:
            win = np.hanning(nfft)
            frames = iq[: nframes * nfft].reshape(nframes, nfft) * win
            spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
            p = np.mean(np.abs(spec) ** 2, axis=0) / (win.sum() ** 2)
            psd = lin2db(p)
        # Receive-only burst view: average the RF power into short time bins.
        # It makes brief transmissions visible without retaining raw I/Q or
        # attempting to recover symbols, bits, or a protocol payload.
        env_bin = max(1, int(round(self.envelope_bin_s * self.fs)))
        env_points = max(1, len(iq) // env_bin)
        usable = env_points * env_bin
        envelope = lin2db(np.mean(np.abs(iq[:usable].reshape(env_points, env_bin)) ** 2,
                                  axis=1))
        with self.psd_lock:
            self.burst_envelope = envelope
            if psd is not None:
                self.psd = psd
                self.psd_seq += 1
        try:
            self.envelope_q.put_nowait((envelope, usable / self.fs))
        except queue.Full:
            # Preserve the newest receive-only display data if the GUI is
            # paused or briefly busy redrawing.
            try:
                self.envelope_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.envelope_q.put_nowait((envelope, usable / self.fs))
            except queue.Full:
                pass
        # ---- raw sample health ----
        re, im = iq.real, iq.imag
        sre, sim = re.std(), im.std()
        self.stats = {
            "rms_db": lin2db(np.mean(re * re + im * im)),
            "clip": 100.0 * np.mean((np.abs(re) > 0.995) | (np.abs(im) > 0.995)),
            "dc": abs(iq.mean()),
            "iq_db": 20 * math.log10(max(sre, 1e-9) / max(sim, 1e-9)),
        }
        # ---- selected audio demodulator ----
        demod, audio = self.demod, self.audio
        if demod is not None and audio is not None:
            audio.push(demod.process(iq))


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
class SelfTest:
    def __init__(self, index, log):
        self.index = index
        self.log = log  # log(level, text); level in INFO/PASS/WARN/FAIL/HEAD
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def res(self, level, msg):
        if level in self.counts:
            self.counts[level] += 1
        self.log(level, msg)

    def run(self):
        self.log("HEAD", "=== RTL-SDR self-test ===")
        try:
            self._run()
        except Exception as exc:
            self.res("FAIL", f"Unexpected error: {exc!r}")
        c = self.counts
        verdict = "FAIL" if c["FAIL"] else ("WARN" if c["WARN"] else "PASS")
        self.log(verdict, f"=== Done: {c['PASS']} passed, {c['WARN']} warnings, {c['FAIL']} failed ===")

    def _run(self):
        # 1. enumeration
        devs = list_devices()
        if not devs:
            self.res("FAIL", "No RTL-SDR device found (driver? cable? another program using it?)")
            return
        self.res("PASS", f"{len(devs)} device(s) found")
        d = devs[self.index]
        self.log("INFO", f"  #{d['index']}: {d['manufacturer']} / {d['product']}  SN {d['serial']}  ({d['name']})")

        # 2. open + tuner
        try:
            sdr = RtlSdr(device_index=self.index)
        except Exception as exc:
            self.res("FAIL", f"Could not open device: {exc}  (close other SDR software)")
            return
        try:
            self._device_tests(sdr)
        finally:
            try:
                _lib.rtlsdr_set_testmode(sdr.dev_p, 0)
            except Exception:
                pass
            sdr.close()

    def _device_tests(self, sdr):
        tuner = sdr.get_tuner_type()
        if tuner == 0:
            self.res("FAIL", "Device opened but tuner chip not recognised")
        else:
            self.res("PASS", f"Device opened, tuner: {TUNER_NAMES.get(tuner, tuner)}")
        gains = sdr.valid_gains_db
        if gains:
            self.res("PASS", f"Gain table: {len(gains)} steps, {gains[0]} .. {gains[-1]} dB")
        else:
            self.res("FAIL", "Tuner returned an empty gain table")

        # 3. USB integrity + throughput (RTL2832 counter test mode)
        self.log("HEAD", "-- USB streaming integrity (counter test mode, 2 s per rate) --")
        for fs in (1.024e6, 2.048e6, 2.4e6, 2.88e6, 3.2e6):
            high = fs > 2.4e6
            try:
                rate, bad, nbytes = self._counter_test(sdr, fs, 2.0)
            except Exception as exc:
                self.res("WARN" if high else "FAIL", f"{fs/1e6:.3f} MS/s: streaming error {exc}")
                continue
            pct = 100 * rate / fs
            msg = f"{fs/1e6:.3f} MS/s: measured {rate/1e6:.3f} MS/s ({pct:.1f}%), {bad} discontinuities in {nbytes/1e6:.1f} MB"
            if bad == 0 and pct > 97:
                self.res("PASS", msg)
            else:
                self.res("WARN" if high else "FAIL",
                         msg + ("  (rates above 2.4 MS/s are often unstable; that's normal)" if high else ""))

        # remaining tests on real RF samples
        sdr.sample_rate = 2.048e6
        lo, hi = TUNER_RANGES.get(tuner, (24, 1766))

        # 4. tuning sweep
        self.log("HEAD", f"-- Tuning sweep ({lo}-{hi} MHz) --")
        apply_gain(sdr, gains[len(gains) // 2] if gains else "auto")
        fails = 0
        for f in (25, 50, 100, 200, 433.92, 868, 1090, 1420, 1700, 2100):
            if not lo <= f <= hi:
                continue
            try:
                sdr.center_freq = f * 1e6
                got = sdr.get_center_freq()
                p, _ = self._measure(sdr)
                ok = abs(got - f * 1e6) < 1e3
                fails += not ok
                self.res("PASS" if ok else "FAIL", f"{f:8.2f} MHz: tuned (readback {got/1e6:.3f} MHz), power {p:6.1f} dBFS")
            except Exception as exc:
                fails += 1
                self.res("FAIL", f"{f:8.2f} MHz: {exc}")
                if not getattr(sdr, "device_opened", True):
                    self.res("FAIL", "Device was closed by the driver after the error; aborting")
                    return

        # 5. gain response
        self.log("HEAD", "-- Gain response @ 100 MHz --")
        sdr.center_freq = 100e6
        if len(gains) >= 3:
            pts = [gains[0], gains[len(gains) // 2], gains[-1]]
            powers = []
            for g in pts:
                apply_gain(sdr, g)
                p, _ = self._measure(sdr)
                powers.append(p)
                self.log("INFO", f"  gain {g:5.1f} dB -> {p:6.1f} dBFS")
            delta = powers[-1] - powers[0]
            if delta >= 6 and powers[1] >= powers[0] - 1:
                self.res("PASS", f"Output power rises with gain (+{delta:.1f} dB from min to max)")
            elif delta >= 2:
                self.res("WARN", f"Weak gain response (+{delta:.1f} dB). Very strong input signal or AGC?")
            else:
                self.res("FAIL", f"No gain response ({delta:+.1f} dB): RF front-end problem")

        # 6. raw sample sanity at a sensible gain
        self.log("HEAD", "-- Raw ADC sample sanity @ 100 MHz --")
        apply_gain(sdr, gains[len(gains) // 2] if gains else "auto")
        _lib.rtlsdr_reset_buffer(sdr.dev_p)
        sdr.read_bytes(65536)
        raw = np.ctypeslib.as_array(sdr.read_bytes(2 * 262144)).copy()
        i_raw, q_raw = raw[0::2].astype(float), raw[1::2].astype(float)
        si, sq = i_raw.std(), q_raw.std()
        distinct = len(np.unique(raw))
        if si < 0.5 or sq < 0.5 or distinct < 8:
            self.res("FAIL", f"ADC looks stuck (std I={si:.2f}, Q={sq:.2f} LSB, {distinct} distinct codes)")
        else:
            self.res("PASS", f"ADC alive: std I={si:.1f}, Q={sq:.1f} LSB, {distinct} distinct codes")
        dci, dcq = i_raw.mean() - 127.5, q_raw.mean() - 127.5
        lvl = "PASS" if max(abs(dci), abs(dcq)) < 5 else "WARN"
        self.res(lvl, f"DC offset: I={dci:+.2f}, Q={dcq:+.2f} LSB")
        imb = 20 * math.log10(max(si, 1e-9) / max(sq, 1e-9))
        self.res("PASS" if abs(imb) < 1 else "WARN", f"I/Q amplitude balance: {imb:+.2f} dB")
        clip = 100 * np.mean((raw == 0) | (raw == 255))
        self.res("PASS" if clip < 1 else "WARN", f"Clipping: {clip:.3f}% of samples"
                 + ("  (reduce gain)" if clip >= 1 else ""))

        self.log("INFO", "Tip: tune to a known FM station with 'FM audio' on to check reception "
                          "and read the frequency error (PPM) from the peak position.")

    def _counter_test(self, sdr, fs, seconds, chunk=1 << 20):
        """RTL2832 test mode outputs an 8-bit counter instead of ADC data.
        Any jump in the sequence (inside a buffer or between consecutive reads)
        means bytes were lost - the same check rtl_test does. Uses the same
        back-to-back synchronous reads as the live stream."""
        sdr.sample_rate = fs
        _lib.rtlsdr_set_testmode(sdr.dev_p, 1)
        try:
            _lib.rtlsdr_reset_buffer(sdr.dev_p)
            sdr.read_bytes(chunk)  # warm-up
            bufs = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:  # read back-to-back, check later
                bufs.append(sdr.read_bytes(chunk))
            dt = time.perf_counter() - t0
        finally:
            # Writing registers right after streaming can fail with a USB pipe
            # error; resetting the endpoint first makes it reliable.
            for _ in range(3):
                _lib.rtlsdr_reset_buffer(sdr.dev_p)
                if _lib.rtlsdr_set_testmode(sdr.dev_p, 0) == 0:
                    break
        a = np.concatenate([np.ctypeslib.as_array(b) for b in bufs])
        bad = int(np.count_nonzero(np.diff(a) != 1))  # uint8 diff wraps mod 256
        return a.size / 2 / dt, bad, a.size

    @staticmethod
    def _measure(sdr, n=131072):
        _lib.rtlsdr_reset_buffer(sdr.dev_p)
        sdr.read_bytes(65536)  # let the tuner / AGC settle
        x = sdr.read_samples(n)
        x = x - x.mean()
        return float(lin2db(np.mean(np.abs(x) ** 2))), x


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # The USB reader thread must get the GIL back quickly after each read,
        # otherwise matplotlib redraws (~70 ms) starve it and samples are lost.
        sys.setswitchinterval(0.0005)
        self.title("RTL-SDR / NooElec Test Bench")
        # Let Tk size the canvas from the actual monitor rather than leaving a
        # large fixed-size plot that can push the controls off a shorter screen.
        self.geometry("1400x820")
        self.minsize(1100, 700)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.stream = None
        self.devices = []
        self.gains = []
        self.log_q = queue.Queue()
        self.selftest_thread = None
        self.avg = None
        self.peak = None
        self.wf = None
        self.ylim = None
        self.last_seq = -1
        self.rate_hist = deque(maxlen=20)
        self.burst_history = deque()
        self.burst_history_end = 0.0
        self._plot_height_ratios = None
        self.tuner_type = None
        self._scroll_canvases = []

        self._build_ui()
        # "zoomed" is the normal Windows maximised state: it retains the title
        # bar and taskbar, unlike a borderless fullscreen window.  Schedule it
        # after the widgets exist so the canvas receives the final dimensions.
        self.after_idle(self._maximize_window)
        self.after(100, self.refresh_devices)
        self.after(50, self._poll)

    def _maximize_window(self):
        """Open maximised where the window manager supports it."""
        try:
            self.state("zoomed")
        except tk.TclError:
            # This fallback keeps the app usable on window managers that do
            # not implement the Windows-specific ``zoomed`` state.
            width = int(self.winfo_screenwidth() * 0.96)
            height = int(self.winfo_screenheight() * 0.90)
            self.geometry(f"{max(1100, width)}x{max(700, height)}+0+0")

    # ------------------------------------------------------------- UI build
    def _build_ui(self):
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        # The controls need more vertical room than a laptop-height screen
        # provides.  Put them in a scrolling viewport instead of clipping the
        # Start button and the lower receiver settings.
        side_view = ttk.Frame(root, width=250)
        side_view.pack(side="left", fill="y", padx=(0, 6))
        side_view.pack_propagate(False)
        self.side_canvas = tk.Canvas(side_view, width=248, highlightthickness=0, borderwidth=0)
        side_scroll = ttk.Scrollbar(side_view, orient="vertical", command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        self.side_canvas.pack(side="left", fill="both", expand=True)
        side_scroll.pack(side="right", fill="y")
        side = ttk.Frame(self.side_canvas)
        self.side_window = self.side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", self._update_side_scrollregion)
        self.side_canvas.bind("<Configure>", self._resize_side_content)
        self._scroll_canvases.append(self.side_canvas)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        main = ttk.Frame(root)
        main.pack(side="left", fill="both", expand=True)

        # device
        f = ttk.LabelFrame(side, text="Device", padding=6)
        f.pack(fill="x", pady=3)
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(f, textvariable=self.dev_var, state="readonly", width=34)
        self.dev_combo.pack(fill="x")
        self.dev_combo.bind("<<ComboboxSelected>>", lambda e: self.probe_device())
        ttk.Button(f, text="Refresh", command=self.refresh_devices).pack(fill="x", pady=(4, 0))
        self.dev_info = ttk.Label(f, text="-", justify="left", wraplength=280)
        self.dev_info.pack(fill="x", pady=(4, 0))

        # Keep the primary action visible, even when the rest of the controls
        # are below the fold on a short screen.
        f = ttk.Frame(side)
        f.pack(fill="x", pady=6)
        self.start_btn = ttk.Button(f, text="▶ Start", command=self.toggle_stream)
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.test_btn = ttk.Button(f, text="Run self-test", command=self.run_selftest)
        self.test_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # tuning
        f = ttk.LabelFrame(side, text="Tuning", padding=6)
        f.pack(fill="x", pady=3)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Freq (MHz)").pack(side="left")
        self.freq_var = tk.StringVar(value="100.000")
        e = ttk.Entry(row, textvariable=self.freq_var, width=12)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.apply_freq())
        ttk.Button(row, text="Set", width=5, command=self.apply_freq).pack(side="left")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        for step in (-1, -0.1, 0.1, 1):
            ttk.Button(row, text=f"{step:+g}", width=6,
                       command=lambda s=step: self.step_freq(s)).pack(side="left", expand=True, fill="x")
        self.preset_var = tk.StringVar(value="Presets...")
        pc = ttk.Combobox(f, textvariable=self.preset_var, state="readonly",
                          values=[p[0] for p in PRESETS])
        pc.pack(fill="x", pady=(4, 0))
        pc.bind("<<ComboboxSelected>>", self.on_preset)

        # radio settings
        f = ttk.LabelFrame(side, text="Receiver", padding=6)
        f.pack(fill="x", pady=3)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Sample rate").grid(row=0, column=0, sticky="w")
        self.rate_var = tk.StringVar(value="2.400 MS/s")
        rc = ttk.Combobox(f, textvariable=self.rate_var, state="readonly", width=12,
                          values=[f"{r/1e6:.3f} MS/s" for r in SAMPLE_RATES])
        rc.grid(row=0, column=1, sticky="ew", pady=2)
        rc.bind("<<ComboboxSelected>>", lambda e: self.on_rate())
        ttk.Label(f, text="FFT size").grid(row=1, column=0, sticky="w")
        self.fft_var = tk.StringVar(value="2048")
        fc = ttk.Combobox(f, textvariable=self.fft_var, state="readonly", width=12,
                          values=[str(n) for n in FFT_SIZES])
        fc.grid(row=1, column=1, sticky="ew", pady=2)
        fc.bind("<<ComboboxSelected>>", lambda e: self.on_fft())
        ttk.Label(f, text="PPM").grid(row=2, column=0, sticky="w")
        self.ppm_var = tk.IntVar(value=0)
        sp = ttk.Spinbox(f, from_=-200, to=200, textvariable=self.ppm_var, width=8, command=self.on_ppm)
        sp.grid(row=2, column=1, sticky="ew", pady=2)
        sp.bind("<Return>", lambda e: self.on_ppm())
        self.agc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Tuner AGC (auto gain)", variable=self.agc_var,
                        command=self.on_gain).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.gain_label = ttk.Label(f, text="Gain: -")
        self.gain_label.grid(row=4, column=0, columnspan=2, sticky="w")
        self.gain_scale = tk.Scale(f, from_=0, to=1, orient="horizontal", showvalue=False,
                                   command=lambda v: self.on_gain())
        self.gain_scale.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.bias_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Bias-tee (4.5 V on antenna port)", variable=self.bias_var,
                        command=self.on_bias).grid(row=6, column=0, columnspan=2, sticky="w")

        # display
        f = ttk.LabelFrame(side, text="Display", padding=6)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="Averaging").pack(anchor="w")
        self.avg_scale = tk.Scale(f, from_=1, to=20, orient="horizontal")
        self.avg_scale.set(4)
        self.avg_scale.pack(fill="x")
        self.peakhold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Peak hold", variable=self.peakhold_var,
                        command=lambda: setattr(self, "peak", None)).pack(anchor="w")

        # time-domain burst display
        f = ttk.LabelFrame(side, text="Burst monitor (receive-only)", padding=6)
        f.pack(fill="x", pady=3)
        self.burst_display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Update burst history", variable=self.burst_display_var,
                        command=self.on_burst_display).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(3, 0))
        ttk.Label(row, text="History (s)").pack(side="left")
        self.burst_span_var = tk.StringVar(value="5")
        span = ttk.Combobox(row, textvariable=self.burst_span_var, state="readonly", width=8,
                            values=("0.25", "0.5", "1", "2", "5", "10", "20"))
        span.pack(side="right")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(3, 0))
        ttk.Label(row, text="Envelope avg. (ms)").pack(side="left")
        self.envelope_bin_ms_var = tk.StringVar(value="0.10")
        bins = ttk.Combobox(row, textvariable=self.envelope_bin_ms_var, state="readonly", width=8,
                            values=("0.10", "0.25", "0.50", "1.00", "2.00", "5.00"))
        bins.pack(side="right")
        bins.bind("<<ComboboxSelected>>", self.on_envelope_resolution)
        ttk.Button(f, text="Reset burst history", command=self.reset_burst_history).pack(fill="x", pady=(3, 0))

        # audio
        f = ttk.LabelFrame(side, text="Audio receiver", padding=6)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="Mode").pack(anchor="w")
        self.audio_mode_var = tk.StringVar(value="WFM broadcast FM")
        self.audio_mode_combo = ttk.Combobox(
            f, textvariable=self.audio_mode_var, state="readonly",
            values=("WFM broadcast FM", "AM airband voice"),
        )
        self.audio_mode_combo.pack(fill="x", pady=(0, 3))
        self.audio_mode_combo.bind("<<ComboboxSelected>>", self.on_audio_mode)
        self.audio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Play demodulated audio", variable=self.audio_var,
                        command=self.on_audio, state="normal" if sd else "disabled").pack(anchor="w")
        ttk.Label(f, text="Volume").pack(anchor="w")
        self.vol_scale = tk.Scale(f, from_=0, to=100, orient="horizontal",
                                  command=lambda v: self.on_volume())
        self.vol_scale.set(50)
        self.vol_scale.pack(fill="x")
        if not sd:
            ttk.Label(f, text="pip install sounddevice to enable", foreground="gray").pack(anchor="w")

        # status
        f = ttk.LabelFrame(side, text="Stream health", padding=6)
        f.pack(fill="x", pady=3)
        self.status_vars = {}
        for key in ("State", "Throughput", "Dropped", "Level", "Clipping", "DC / IQ", "Peak", "Audio"):
            row = ttk.Frame(f)
            row.pack(fill="x")
            ttk.Label(row, text=key + ":", width=11).pack(side="left")
            v = tk.StringVar(value="-")
            ttk.Label(row, textvariable=v).pack(side="left")
            self.status_vars[key] = v

        # plots
        # A compact requested size prevents the plot from dictating a tall
        # initial window.  TkAgg expands it to the space available afterwards.
        self.fig = Figure(figsize=(8, 4.8), dpi=100, facecolor="#1b1d22")
        self.plot_grid = self.fig.add_gridspec(3, 1, height_ratios=[0.65, 0.65, 1.2], hspace=0.10)
        self.ax_spec = self.fig.add_subplot(self.plot_grid[0])
        self.ax_burst = self.fig.add_subplot(self.plot_grid[1])
        self.ax_wf = self.fig.add_subplot(self.plot_grid[2], sharex=self.ax_spec)
        for ax in (self.ax_spec, self.ax_burst, self.ax_wf):
            ax.set_facecolor("#111317")
            ax.tick_params(colors="#c8ccd4", labelsize=8)
            for s in ax.spines.values():
                s.set_color("#444a55")
        self.ax_spec.grid(True, color="#2c313a", lw=0.6)
        self.ax_spec.set_ylabel("dBFS", color="#c8ccd4")
        self.ax_spec.tick_params(labelbottom=False)
        self.ax_burst.grid(True, color="#2c313a", lw=0.6)
        self.ax_burst.set_ylabel("Power (dBFS)", color="#c8ccd4")
        self.ax_burst.set_xlabel("Burst history (s; now at right) — receive-only", color="#c8ccd4")
        self.ax_wf.set_xlabel("Frequency (MHz)  -  click to tune", color="#c8ccd4")
        self.ax_wf.set_yticks([])
        self.line, = self.ax_spec.plot([], [], color="#4fc3f7", lw=0.9)
        self.peak_line, = self.ax_spec.plot([], [], color="#ffb74d", lw=0.7, alpha=0.8)
        self.burst_line, = self.ax_burst.plot([], [], color="#b9f6ca", lw=0.9)
        self.center_line = self.ax_spec.axvline(0, color="#ef5350", lw=0.6, alpha=0.6)
        self.wf_img = self.ax_wf.imshow(np.zeros((WF_ROWS, 2)), aspect="auto", cmap="turbo",
                                        origin="upper", interpolation="nearest")
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.07)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)
        spectrum_tab = ttk.Frame(self.notebook)
        params_tab = ttk.Frame(self.notebook)
        self.notebook.add(spectrum_tab, text="Spectrum")
        self.notebook.add(params_tab, text="Radio parameters")
        self._build_params_tab(params_tab)
        self.canvas = FigureCanvasTkAgg(self.fig, master=spectrum_tab)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        self.canvas_widget.bind("<Configure>", self._resize_plot)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # log
        lf = ttk.LabelFrame(main, text="Log / self-test results", padding=4)
        lf.pack(fill="x", pady=(6, 0))
        self.log_text = tk.Text(lf, height=12, font=("Consolas", 9), wrap="none",
                                bg="#15171b", fg="#d0d4dc", insertbackground="#d0d4dc")
        sb = ttk.Scrollbar(lf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        for tag, col in (("PASS", "#81c784"), ("FAIL", "#ef5350"), ("WARN", "#ffb74d"),
                         ("HEAD", "#64b5f6"), ("INFO", "#d0d4dc")):
            self.log_text.tag_configure(tag, foreground=col)

    def _resize_plot(self, event):
        """Keep the waterfall useful when a maximised window is short."""
        ratios = (0.45, 0.55, 1.5) if event.height < 500 else (0.65, 0.65, 1.2)
        if ratios != self._plot_height_ratios:
            self.plot_grid.set_height_ratios(ratios)
            self._plot_height_ratios = ratios
            self.canvas.draw_idle()

    def _update_side_scrollregion(self, _event=None):
        """Expose all controls when their requested height exceeds the window."""
        bounds = self.side_canvas.bbox("all")
        if bounds:
            self.side_canvas.configure(scrollregion=bounds)

    def _resize_side_content(self, event):
        self.side_canvas.itemconfigure(self.side_window, width=event.width)
        self._update_side_scrollregion()

    def _on_mousewheel(self, event):
        """Scroll whichever scrollable panel is under the pointer."""
        if not event.delta:
            return
        try:
            w = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):  # e.g. pointer over a combobox popdown
            return
        while w is not None:
            if isinstance(w, tk.Text):
                return  # text boxes scroll themselves
            if w in self._scroll_canvases:
                w.yview_scroll(-int(event.delta / 120), "units")
                return
            w = w.master

    def _scrollable(self, parent):
        """A vertically scrolling frame filling parent."""
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas, padding=4)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self._scroll_canvases.append(canvas)
        return inner

    def on_burst_display(self):
        state = "resumed" if self.burst_display_var.get() else "paused"
        self.log("INFO", f"Burst-history display {state}")

    def reset_burst_history(self):
        """Clear only the safe, power-versus-time display history."""
        self.burst_history.clear()
        self.burst_history_end = 0.0
        if self.stream:
            while True:
                try:
                    self.stream.envelope_q.get_nowait()
                except queue.Empty:
                    break
        self.burst_line.set_data([], [])
        self.canvas.draw_idle()

    def on_envelope_resolution(self, _event=None):
        """Change only the power-graph averaging, never the SDR's raw capture."""
        try:
            bin_s = float(self.envelope_bin_ms_var.get()) / 1000.0
        except ValueError:
            return
        if self.stream:
            self.stream.envelope_bin_s = bin_s
        self.reset_burst_history()
        self.log("INFO", f"Burst envelope averaging set to {bin_s * 1000:g} ms")

    # ------------------------------------------------------------- radio parameters tab
    # Every control goes straight to librtlsdr. While streaming, the call is
    # queued to the USB reader thread (the only thread allowed to touch the
    # device). Settings are also remembered in self.adv and re-applied on every
    # stream start, so they survive sample-rate changes and Stop/Start.
    def _build_params_tab(self, parent):
        self.adv = {}  # key -> (description, fn(sdr)), in the order last set
        self.adv_q = queue.Queue()  # results from device jobs -> GUI thread
        self.adv_caps = {"r82xx": [], "e4k": [], "zeroif": []}
        self.adv_last_auto = 0.0
        inner = self._scrollable(parent)
        cols = []
        for c in range(3):
            inner.columnconfigure(c, weight=1, uniform="adv")
            col = ttk.Frame(inner)
            col.grid(row=0, column=c, sticky="new", padx=3)
            cols.append(col)
        self._build_adv_readback(cols[0])
        self._build_adv_tuner(cols[1])
        self._build_adv_rtl(cols[2])
        self._build_adv_eeprom(inner)

    @staticmethod
    def _hint(parent, text, **grid):
        lbl = ttk.Label(parent, text=text, foreground="gray", wraplength=320, justify="left")
        if grid:
            lbl.grid(**grid)
        else:
            lbl.pack(anchor="w", pady=(3, 0))
        return lbl

    @staticmethod
    def _mono_text(parent, height, width):
        return tk.Text(parent, height=height, width=width, font=("Consolas", 9), wrap="none",
                       bg="#15171b", fg="#d0d4dc", insertbackground="#d0d4dc")

    def _build_adv_readback(self, col):
        f = ttk.LabelFrame(col, text="Device readback", padding=6)
        f.pack(fill="x", pady=3)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Read now", command=self.adv_readback).pack(side="left")
        self.adv_auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Auto every 2 s", variable=self.adv_auto_var).pack(side="left", padx=8)
        grid = ttk.Frame(f)
        grid.pack(fill="x", pady=(4, 0))
        self.rb_vars = {}
        keys = ("Library", "Source", "Tuner", "Center freq", "Sample rate", "Freq. correction",
                "Tuner gain", "RTL xtal", "Tuner xtal", "Direct sampling", "Offset tuning",
                "Tuner PLL", "GPIO levels")
        for i, key in enumerate(keys):
            ttk.Label(grid, text=key + ":").grid(row=i, column=0, sticky="nw")
            v = tk.StringVar(value="-")
            ttk.Label(grid, textvariable=v, wraplength=230, justify="left").grid(
                row=i, column=1, sticky="w", padx=(6, 0))
            self.rb_vars[key] = v
        self.rb_vars["Library"].set(lib_version() if _lib else "not loaded")
        self._hint(f, "Readback goes through the USB control pipe; while streaming each read "
                      "causes a short gap in the samples.")

        f = ttk.LabelFrame(col, text="Tuner I2C registers", padding=6)
        f.pack(fill="x", pady=3)
        self.i2c_text = self._mono_text(f, 4, 28)
        self.i2c_text.pack(fill="x")
        self.i2c_text.insert("end", "(press Read now)")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        self.i2c_vars = {}
        for name, default in (("Reg", "05"), ("Mask", "FF"), ("Value", "00")):
            ttk.Label(row, text=name).pack(side="left")
            v = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=v, width=4).pack(side="left", padx=(2, 6))
            self.i2c_vars[name] = v
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        for text, mode in (("Write once", "once"), ("Override", "override"), ("Clear override", "clear")):
            ttk.Button(row, text=text, command=lambda m=mode: self.adv_i2c(m)).pack(
                side="left", expand=True, fill="x")
        self._hint(f, "Hex values. 'Write once' changes the bits selected by Mask; the driver may "
                      "rewrite the register on the next retune. 'Override' forces those bits "
                      "permanently (re-applied on every start). R820T/2: 0x00-0x04 are read-only.")

    def _build_adv_tuner(self, col):
        f = ttk.LabelFrame(col, text="Tuner gain", padding=6)
        f.pack(fill="x", pady=3)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="Gain mode").grid(row=0, column=0, sticky="w")
        self.adv_gainmode_var = tk.StringVar(value="manual")
        rb = ttk.Frame(f)
        rb.grid(row=0, column=1, sticky="w")
        for text, val in (("Manual", "manual"), ("Tuner AGC", "auto")):
            ttk.Radiobutton(rb, text=text, value=val, variable=self.adv_gainmode_var,
                            command=self.adv_tuner_gain).pack(side="left", padx=(0, 6))
        self.adv_stage = {}
        for i, (name, default) in enumerate((("LNA", 8), ("Mixer", 8), ("VGA", 11)), start=1):
            ttk.Label(f, text=f"{name} index").grid(row=i, column=0, sticky="w")
            v = tk.IntVar(value=default)
            # Apply on release only: a Scale's command also fires on
            # programmatic changes, which must not reach the device.
            s = tk.Scale(f, from_=0, to=15, orient="horizontal", variable=v)
            s.grid(row=i, column=1, sticky="ew")
            s.bind("<ButtonRelease-1>", lambda e, s=s: s.cget("state") == "normal"
                   and self.after_idle(self.adv_gain_stages))
            self.adv_stage[name] = v
            self.adv_caps["r82xx"].append(s)
        self.adv_vga_auto_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(f, text="VGA driven by the RTL2832 AGC loop", variable=self.adv_vga_auto_var,
                             command=self.adv_gain_stages)
        cb.grid(row=4, column=0, columnspan=2, sticky="w")
        self.adv_caps["r82xx"].append(cb)
        self._hint(f, "R820T/2 gain-table indices 0-15 (moving one selects Manual). "
                      "The sidebar gain slider overrides these.", row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(f, text="IF/VGA mode").grid(row=6, column=0, sticky="w", pady=(6, 0))
        row = ttk.Frame(f)
        row.grid(row=6, column=1, sticky="ew", pady=(6, 0))
        self.adv_ifmode_var = tk.StringVar(value="10011")
        e = ttk.Entry(row, textvariable=self.adv_ifmode_var, width=8)
        e.pack(side="left")
        b = ttk.Button(row, text="Set", width=5, command=self.adv_if_mode)
        b.pack(side="left", padx=4)
        self.adv_caps["r82xx"] += [e, b]
        self._hint(f, "0 = VGA from RTL2832 AGC; -2500..2500 = fixed gain in 0.1 dB; "
                      "10000+i = VGA index i (0-15); 10016+i = index i with AGC. Default 10011.",
                   row=7, column=0, columnspan=2, sticky="w")

        ttk.Label(f, text="E4000 IF stage").grid(row=8, column=0, sticky="w", pady=(6, 0))
        row = ttk.Frame(f)
        row.grid(row=8, column=1, sticky="ew", pady=(6, 0))
        self.adv_ifstage_var = tk.IntVar(value=1)
        self.adv_ifgain_var = tk.StringVar(value="0")
        sp = ttk.Spinbox(row, from_=1, to=6, textvariable=self.adv_ifstage_var, width=3)
        sp.pack(side="left")
        e = ttk.Entry(row, textvariable=self.adv_ifgain_var, width=6)
        e.pack(side="left", padx=4)
        ttk.Label(row, text="dB").pack(side="left")
        b = ttk.Button(row, text="Set", width=5, command=self.adv_if_gain)
        b.pack(side="left", padx=4)
        self.adv_caps["e4k"] += [sp, e, b]

        f = ttk.LabelFrame(col, text="Tuner filter and mixer", padding=6)
        f.pack(fill="x", pady=3)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="IF bandwidth").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=0, column=1, sticky="ew")
        self.adv_bw_var = tk.StringVar(value="Auto")
        cbx = ttk.Combobox(row, textvariable=self.adv_bw_var, width=10,
                           values=["Auto"] + [f"{k} kHz" for k in BANDWIDTHS_KHZ])
        cbx.pack(side="left")
        cbx.bind("<<ComboboxSelected>>", lambda e: self.adv_bandwidth())
        cbx.bind("<Return>", lambda e: self.adv_bandwidth())
        ttk.Button(row, text="Set", width=5, command=self.adv_bandwidth).pack(side="left", padx=4)
        self.adv_bw_applied = tk.StringVar(value="applied: -")
        ttk.Label(f, textvariable=self.adv_bw_applied).grid(row=1, column=1, sticky="w")

        ttk.Label(f, text="Band center (Hz)").grid(row=2, column=0, sticky="w", pady=(4, 0))
        row = ttk.Frame(f)
        row.grid(row=2, column=1, sticky="ew", pady=(4, 0))
        self.adv_bc_var = tk.StringVar(value="0")
        e = ttk.Entry(row, textvariable=self.adv_bc_var, width=10)
        e.pack(side="left")
        b = ttk.Button(row, text="Set", width=5, command=self.adv_band_center)
        b.pack(side="left", padx=4)
        self.adv_caps["r82xx"] += [e, b]

        ttk.Label(f, text="Mixer sideband").grid(row=3, column=0, sticky="w", pady=(4, 0))
        row = ttk.Frame(f)
        row.grid(row=3, column=1, sticky="w", pady=(4, 0))
        self.adv_sb_var = tk.IntVar(value=0)
        for text, val in (("Lower", 0), ("Upper", 1)):
            r = ttk.Radiobutton(row, text=text, value=val, variable=self.adv_sb_var, command=self.adv_sideband)
            r.pack(side="left", padx=(0, 6))
            self.adv_caps["r82xx"].append(r)

        ttk.Label(f, text="Harmonic RX").grid(row=4, column=0, sticky="w", pady=(4, 0))
        row = ttk.Frame(f)
        row.grid(row=4, column=1, sticky="ew", pady=(4, 0))
        self.adv_harm_var = tk.IntVar(value=0)
        sp = ttk.Spinbox(row, from_=0, to=16, textvariable=self.adv_harm_var, width=4)
        sp.pack(side="left")
        b = ttk.Button(row, text="Set", width=5, command=self.adv_harmonic)
        b.pack(side="left", padx=4)
        self.adv_caps["r82xx"] += [sp, b]

        self.adv_dither_var = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(f, text="PLL dithering (restarts the stream)", variable=self.adv_dither_var,
                             command=self.adv_dithering)
        cb.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.adv_caps["r82xx"].append(cb)
        self._hint(f, "Band center shifts the tuner's passband inside the sampled band. Harmonic RX "
                      "(0 = off) uses the Nth PLL harmonic to reach above ~1.76 GHz on R820T/2. "
                      "Disable dithering for phase-coherent multi-dongle setups.",
                   row=6, column=0, columnspan=2, sticky="w")

    def _build_adv_rtl(self, col):
        f = ttk.LabelFrame(col, text="RTL2832U demodulator", padding=6)
        f.pack(fill="x", pady=3)
        f.columnconfigure(1, weight=1)
        self.adv_dagc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Digital AGC (after the ADC)", variable=self.adv_dagc_var,
                        command=self.adv_digital_agc).grid(row=0, column=0, columnspan=2, sticky="w")
        self.adv_offset_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(f, text="Offset tuning (zero-IF tuners only)", variable=self.adv_offset_var,
                             command=self.adv_offset_tuning)
        cb.grid(row=1, column=0, columnspan=2, sticky="w")
        self.adv_caps["zeroif"].append(cb)
        self.adv_testmode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Counter test mode (outputs a counter, not RF)",
                        variable=self.adv_testmode_var,
                        command=self.adv_testmode).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(f, text="Direct sampling").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.adv_ds_var = tk.StringVar(value=DS_MODES[0][0])
        ttk.Combobox(f, textvariable=self.adv_ds_var, state="readonly",
                     values=[m[0] for m in DS_MODES]).grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(f, text="Threshold (MHz)").grid(row=4, column=0, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=4, column=1, sticky="ew")
        self.adv_ds_thr_var = tk.StringVar(value="24")
        ttk.Entry(row, textvariable=self.adv_ds_thr_var, width=8).pack(side="left")
        ttk.Button(row, text="Apply", command=self.adv_direct_sampling).pack(side="left", padx=4)
        self._hint(f, "Direct sampling bypasses the tuner and feeds the antenna to the ADC (HF "
                      "reception, needs a dongle wired for it). Threshold modes switch "
                      "automatically below the threshold frequency.", row=5, column=0, columnspan=2,
                   sticky="w")

        ttk.Label(f, text="RTL xtal (Hz)").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.adv_xtal_rtl_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.adv_xtal_rtl_var, width=12).grid(row=6, column=1, sticky="w", pady=(6, 0))
        ttk.Label(f, text="Tuner xtal (Hz)").grid(row=7, column=0, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=7, column=1, sticky="ew")
        self.adv_xtal_tuner_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.adv_xtal_tuner_var, width=12).pack(side="left")
        ttk.Button(row, text="Set", width=5, command=self.adv_xtal).pack(side="left", padx=4)
        self._hint(f, "Tells the driver the real crystal frequencies (default 28.8 MHz). Only change "
                      "them for a modified dongle or a precise calibration; prefer PPM otherwise.",
                   row=8, column=0, columnspan=2, sticky="w")

        f = ttk.LabelFrame(col, text="GPIO pins", padding=6)
        f.pack(fill="x", pady=3)
        self.gpio_vars, self.gpio_levels = {}, {}
        for pin in range(8):
            ttk.Label(f, text=f"GPIO {pin}").grid(row=pin, column=0, sticky="w")
            v = tk.StringVar(value="Untouched")
            cbx = ttk.Combobox(f, textvariable=v, state="readonly", width=11,
                               values=("Untouched", "Input", "Output low", "Output high"))
            cbx.grid(row=pin, column=1, sticky="w", padx=4, pady=1)
            cbx.bind("<<ComboboxSelected>>", lambda e, p=pin: self.adv_gpio(p))
            lv = tk.StringVar(value="?")
            ttk.Label(f, textvariable=lv, width=2).grid(row=pin, column=2)
            ttk.Label(f, text=GPIO_NOTES.get(pin, ""), foreground="gray").grid(row=pin, column=3, sticky="w")
            self.gpio_vars[pin], self.gpio_levels[pin] = v, lv
        self._hint(f, "Level column: last readback. Stopping the stream switches GPIO 0 (bias-tee) off.",
                   row=8, column=0, columnspan=4, sticky="w")

        f = ttk.LabelFrame(col, text="Driver option string", padding=6)
        f.pack(fill="x", pady=3)
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.adv_opt_var = tk.StringVar(value="")
        e = ttk.Entry(row, textvariable=self.adv_opt_var)
        e.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev: self.adv_opt_string())
        ttk.Button(row, text="Apply", width=6, command=self.adv_opt_string).pack(side="left", padx=4)
        ttk.Button(row, text="Help", width=5, command=self.adv_opt_help).pack(side="left")
        self._hint(f, "Raw librtlsdr options, e.g. vcocmax=5:vcoalgo=1 (VCO tweaks and anything "
                      "else the driver parses).")

    def _build_adv_eeprom(self, inner):
        f = ttk.LabelFrame(inner, text="EEPROM (read-only)", padding=6)
        f.grid(row=1, column=0, columnspan=3, sticky="ew", padx=3, pady=3)
        f.columnconfigure(0, weight=1)
        ttk.Button(f, text="Read EEPROM", command=self.adv_read_eeprom).grid(row=0, column=0, sticky="w")
        self.eeprom_text = self._mono_text(f, 8, 74)
        self.eeprom_text.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        sb = ttk.Scrollbar(f, command=self.eeprom_text.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=(4, 0))
        self.eeprom_text.configure(yscrollcommand=sb.set)
        self.eeprom_info = ttk.Label(f, text="", justify="left")
        self.eeprom_info.grid(row=1, column=2, sticky="nw", padx=(10, 0), pady=(4, 0))
        self._hint(f, "Writing the EEPROM is deliberately not offered: a bad write can change the "
                      "USB IDs so Windows no longer binds the WinUSB driver. Use rtl_eeprom if you "
                      "really need to change it.", row=2, column=0, columnspan=3, sticky="w")

    # -- device access -------------------------------------------------------
    def _adv_set(self, key, desc, fn, persist=True):
        """Run fn(sdr) on the device now (if streaming) and remember it for later starts."""
        if persist:
            self.adv.pop(key, None)
            self.adv[key] = (desc, fn)
        if self.stream:
            def run(sdr):
                fn(sdr)
                self.log("INFO", f"Set {desc}")
            self.stream.control(key, run, self.stream.sdr)
            self.adv_readback()
        elif persist:
            self.log("INFO", f"{desc}: stored, applied when the stream starts")
        else:
            self.log("WARN", f"{desc}: start the stream first (needs the device open)")

    def _adv_job(self, key, job):
        """Run job(sdr) -> (kind, data) on the device and deliver the result to _adv_poll.
        Uses the stream's reader thread when streaming, else a temporary handle."""
        if self.stream:
            self.stream.control(key, lambda sdr: self.adv_q.put(job(sdr)), self.stream.sdr)
            return
        if self.busy() or not self.devices:
            self.log("WARN", f"{key}: no device available (self-test running or none found)")
            return
        try:
            sdr = RtlSdr(device_index=self.dev_index(), dithering_enabled=self.adv_dither_var.get())
        except Exception as exc:
            self.log("FAIL", f"{key}: cannot open device: {exc}")
            return
        try:
            self.adv_q.put(job(sdr))
        except Exception as exc:
            self.log("FAIL", f"{key} failed: {exc}")
        finally:
            sdr.close()

    def adv_readback(self):
        streaming = self.stream is not None
        self.adv_last_auto = time.perf_counter()

        def job(sdr):
            st = read_radio_state(sdr)
            st["streaming"] = streaming
            return "state", st
        self._adv_job("Readback", job)

    def adv_read_eeprom(self):
        self._adv_job("EEPROM read", lambda sdr: ("eeprom", read_eeprom(sdr)))

    def _adv_apply_all(self):
        """Queue every remembered setting on a freshly opened stream."""
        for key, (_desc, fn) in self.adv.items():
            self.stream.control(key, fn, self.stream.sdr)
        if self.adv:
            self.log("INFO", "Re-applying radio parameters: " + ", ".join(d for d, _ in self.adv.values()))
        self.adv_readback()

    def _adv_forget(self, *keys):
        for k in keys:
            self.adv.pop(k, None)

    def _adv_update_caps(self):
        t = self.tuner_type
        on = {"r82xx": t in R82XX_TUNERS, "e4k": t == 1, "zeroif": t in ZERO_IF_TUNERS}
        for cap, widgets in self.adv_caps.items():
            for w in widgets:
                if isinstance(w, ttk.Widget):
                    w.state(["!disabled"] if on[cap] else ["disabled"])
                else:
                    w.configure(state="normal" if on[cap] else "disabled")

    # -- results ---------------------------------------------------------------
    def _adv_poll(self):
        while True:
            try:
                kind, data = self.adv_q.get_nowait()
            except queue.Empty:
                break
            if kind == "state":
                self._adv_show_state(data)
            elif kind == "eeprom":
                self.eeprom_text.delete("1.0", "end")
                self.eeprom_text.insert("end", hexdump(data))
                self.eeprom_info.config(text="\n".join(decode_eeprom(data)))
                self.log("INFO", "EEPROM read (256 bytes)")
            elif kind == "bw":
                self.adv_bw_applied.set(f"applied: {data / 1e3:g} kHz" if data else "applied: unknown")
        if (self.stream and self.adv_auto_var.get()
                and time.perf_counter() - self.adv_last_auto >= 2.0):
            self.adv_readback()

    def _adv_show_state(self, st):
        def val(key, fmt):
            x = st.get(key)
            if isinstance(x, Exception):
                return f"n/a ({x})"
            return fmt(x)

        v = self.rb_vars
        v["Source"].set("live stream" if st["streaming"]
                        else "idle: fresh open, driver defaults (start the stream to see your settings)")
        v["Tuner"].set(val("tuner", lambda t: TUNER_NAMES.get(t, f"type {t}")))
        v["Center freq"].set(val("center_freq", lambda f: f"{f / 1e6:.6f} MHz"))
        v["Sample rate"].set(val("sample_rate", lambda r: f"{r / 1e6:.6f} MS/s"))
        v["Freq. correction"].set(val("ppm", lambda p: f"{p} ppm"))
        v["Tuner gain"].set(val("tuner_gain", lambda g: f"{g:.1f} dB"))
        v["RTL xtal"].set(val("xtal", lambda x: f"{x[0]:,} Hz"))
        v["Tuner xtal"].set(val("xtal", lambda x: f"{x[1]:,} Hz"))
        v["Direct sampling"].set(val("direct_sampling", lambda d: {0: "off", 1: "I branch", 2: "Q branch"}.get(d, d)))
        v["Offset tuning"].set(val("offset_tuning", lambda o: "on" if o else "off"))
        v["Tuner PLL"].set(val("pll", lambda p: {0: "locked", 1: "NOT locked",
                                                 -2: "not reported by this tuner"}.get(p, f"error {p}")))
        v["GPIO levels"].set(val("gpio", lambda g: f"0b{g:08b}  (GPIO 7..0)"))
        xt = st.get("xtal")
        if isinstance(xt, tuple):
            if not self.adv_xtal_rtl_var.get():
                self.adv_xtal_rtl_var.set(str(xt[0]))
            if not self.adv_xtal_tuner_var.get():
                self.adv_xtal_tuner_var.set(str(xt[1]))
        g = st.get("gpio")
        for pin, lv in self.gpio_levels.items():
            lv.set("?" if isinstance(g, Exception) else str((g >> pin) & 1))
        regs = st.get("i2c")
        self.i2c_text.delete("1.0", "end")
        self.i2c_text.insert("end", hexdump(regs, 8, ascii_col=False) if isinstance(regs, bytes)
                             else f"not readable: {regs}")

    # -- controls ----------------------------------------------------------------
    def adv_tuner_gain(self):
        """Tuner gain mode; Manual on an R820T/2 also applies the LNA/Mixer/VGA indices."""
        if self.adv_gainmode_var.get() == "auto":
            self._adv_set("Tuner gain", "tuner AGC on",
                          lambda sdr: chk("rtlsdr_set_tuner_gain_mode",
                                          _lib.rtlsdr_set_tuner_gain_mode(sdr.dev_p, 0)))
        elif self.tuner_type in R82XX_TUNERS:
            self.adv_gain_stages()
        else:
            self._adv_set("Tuner gain", "tuner gain manual",
                          lambda sdr: chk("rtlsdr_set_tuner_gain_mode",
                                          _lib.rtlsdr_set_tuner_gain_mode(sdr.dev_p, 1)))

    def adv_gain_stages(self):
        lna, mix = self.adv_stage["LNA"].get(), self.adv_stage["Mixer"].get()
        vga = 16 if self.adv_vga_auto_var.get() else self.adv_stage["VGA"].get()
        self.adv_gainmode_var.set("manual")

        def fn(sdr):
            chk("rtlsdr_set_tuner_gain_mode", _lib.rtlsdr_set_tuner_gain_mode(sdr.dev_p, 1))
            xcall("rtlsdr_set_tuner_gain_ext", sdr.dev_p, lna, mix, vga)
        self._adv_set("Tuner gain", f"gain stages LNA {lna}, mixer {mix}, VGA {'AGC' if vga == 16 else vga}", fn)

    def adv_if_mode(self):
        try:
            mode = int(self.adv_ifmode_var.get())
        except ValueError:
            self.log("WARN", "IF mode must be an integer")
            return
        self._adv_set("IF mode", f"IF/VGA mode {mode}",
                      lambda sdr: xcall("rtlsdr_set_tuner_if_mode", sdr.dev_p, mode))

    def adv_if_gain(self):
        try:
            stage = int(self.adv_ifstage_var.get())
            tenths = round(float(self.adv_ifgain_var.get().replace(",", ".")) * 10)
        except (ValueError, tk.TclError):
            self.log("WARN", "IF stage must be 1-6 and gain a number in dB")
            return
        self._adv_set(f"IF gain {stage}", f"IF stage {stage} gain {tenths / 10:g} dB",
                      lambda sdr: xcall("rtlsdr_set_tuner_if_gain", sdr.dev_p, stage, tenths))

    def adv_bandwidth(self):
        text = self.adv_bw_var.get().strip().lower()
        try:
            bw = 0 if text in ("", "auto", "0") else round(float(text.replace("khz", "").strip()) * 1e3)
        except ValueError:
            self.log("WARN", "Bandwidth: pick a value or type it in kHz")
            return

        def fn(sdr):
            applied = ctypes.c_uint32()
            xcall("rtlsdr_set_and_get_tuner_bandwidth", sdr.dev_p, bw, ctypes.byref(applied), 1)
            self.adv_q.put(("bw", applied.value))
        self._adv_set("Bandwidth", f"IF bandwidth {'auto' if bw == 0 else f'{bw / 1e3:g} kHz'}", fn)

    def adv_band_center(self):
        try:
            hz = int(float(self.adv_bc_var.get()))
        except ValueError:
            self.log("WARN", "Band center must be a number in Hz")
            return
        self._adv_set("Band center", f"band center {hz} Hz",
                      lambda sdr: xcall("rtlsdr_set_tuner_band_center", sdr.dev_p, hz))

    def adv_sideband(self):
        sb = self.adv_sb_var.get()
        self._adv_set("Sideband", f"mixer sideband {'upper' if sb else 'lower'}",
                      lambda sdr: xcall("rtlsdr_set_tuner_sideband", sdr.dev_p, sb))

    def adv_harmonic(self):
        try:
            n = int(self.adv_harm_var.get())
        except (ValueError, tk.TclError):
            return
        self._adv_set("Harmonic", f"harmonic RX {n or 'off'}",
                      lambda sdr: xcall("rtlsdr_set_harmonic_rx", sdr.dev_p, n))

    def adv_dithering(self):
        self.log("INFO", f"PLL dithering {'on' if self.adv_dither_var.get() else 'off'}")
        if self.stream:  # only settable before the first tune: reopen
            self.stop_stream()
            self.start_stream()

    def adv_digital_agc(self):
        on = int(self.adv_dagc_var.get())
        self._adv_set("Digital AGC", f"RTL2832 digital AGC {'on' if on else 'off'}",
                      lambda sdr: chk("rtlsdr_set_agc_mode", _lib.rtlsdr_set_agc_mode(sdr.dev_p, on)))

    def adv_offset_tuning(self):
        on = int(self.adv_offset_var.get())
        self._adv_set("Offset tuning", f"offset tuning {'on' if on else 'off'}",
                      lambda sdr: xcall("rtlsdr_set_offset_tuning", sdr.dev_p, on))

    def adv_testmode(self):
        on = int(self.adv_testmode_var.get())
        self._adv_set("Test mode", f"counter test mode {'on' if on else 'off'}",
                      lambda sdr: chk("rtlsdr_set_testmode", _lib.rtlsdr_set_testmode(sdr.dev_p, on)))

    def adv_direct_sampling(self):
        mode = dict(DS_MODES)[self.adv_ds_var.get()]
        try:
            thr = max(0, round(float(self.adv_ds_thr_var.get().replace(",", ".")) * 1e6))
        except ValueError:
            self.log("WARN", "Threshold must be a frequency in MHz")
            return

        def fn(sdr):
            xcall("rtlsdr_set_ds_mode", sdr.dev_p, mode, thr)
            if mode <= 2:
                chk("rtlsdr_set_direct_sampling", _lib.rtlsdr_set_direct_sampling(sdr.dev_p, mode))
        desc = f"direct sampling: {self.adv_ds_var.get()}" + (f" ({thr / 1e6:g} MHz)" if mode > 2 else "")
        self._adv_set("Direct sampling", desc, fn)

    def adv_xtal(self):
        try:
            rtl = int(float(self.adv_xtal_rtl_var.get() or 0))
            tun = int(float(self.adv_xtal_tuner_var.get() or 0))
        except ValueError:
            self.log("WARN", "Crystal frequencies must be numbers in Hz")
            return
        self._adv_set("Xtal", f"xtal RTL {rtl} Hz, tuner {tun} Hz",
                      lambda sdr: chk("rtlsdr_set_xtal_freq", _lib.rtlsdr_set_xtal_freq(sdr.dev_p, rtl, tun)))

    def adv_gpio(self, pin):
        mode = self.gpio_vars[pin].get()
        key = f"GPIO {pin}"
        if mode == "Untouched":
            self._adv_forget(key)
            self.log("INFO", f"GPIO {pin} will no longer be set on start (current level unchanged)")
            return
        if mode == "Output high" and pin == 0 and not messagebox.askyesno(
                "GPIO 0", "On most NooElec / v3-style dongles GPIO 0 switches ~4.5 V onto the antenna "
                          "connector (bias-tee).\n\nOnly continue if your antenna or LNA expects it."):
            self.gpio_vars[pin].set("Untouched")
            return
        if pin in (4, 6) and not messagebox.askyesno(
                f"GPIO {pin}", f"GPIO {pin} is used by the hardware ({GPIO_NOTES[pin]}). "
                               "Changing it can stop reception until the dongle is reopened.\n\nContinue?"):
            self.gpio_vars[pin].set("Untouched")
            return

        def fn(sdr):
            d = sdr.dev_p
            if mode == "Input":
                chk("rtlsdr_set_gpio_input", _lib.rtlsdr_set_gpio_input(d, pin))
            else:
                chk("rtlsdr_set_gpio_output", _lib.rtlsdr_set_gpio_output(d, pin))
                chk("rtlsdr_set_gpio_bit", _lib.rtlsdr_set_gpio_bit(d, pin, int(mode == "Output high")))
        self._adv_set(key, f"GPIO {pin} = {mode.lower()}", fn)

    def adv_i2c(self, mode):
        try:
            reg, mask, value = (int(self.i2c_vars[k].get(), 16) for k in ("Reg", "Mask", "Value"))
            if not (0 <= reg <= 0xFF and 0 <= mask <= 0xFF and 0 <= value <= 0xFF):
                raise ValueError
        except ValueError:
            self.log("WARN", "I2C register, mask and value must be hex bytes (00-FF)")
            return
        name = "rtlsdr_set_tuner_i2c_override" if mode != "once" else "rtlsdr_set_tuner_i2c_register"
        if mode == "clear":
            self._adv_forget(f"I2C override {reg:02X}")
            self._adv_set(f"I2C clear {reg:02X}", f"tuner reg 0x{reg:02X} override cleared",
                          lambda sdr: xcall(name, sdr.dev_p, reg, 0xFF, 0x100), persist=False)
        elif mode == "override":
            self._adv_set(f"I2C override {reg:02X}",
                          f"tuner reg 0x{reg:02X} override mask 0x{mask:02X} = 0x{value:02X}",
                          lambda sdr: xcall(name, sdr.dev_p, reg, mask, value))
        else:
            self._adv_set(f"I2C {reg:02X}", f"tuner reg 0x{reg:02X} mask 0x{mask:02X} = 0x{value:02X}",
                          lambda sdr: xcall(name, sdr.dev_p, reg, mask, value), persist=False)

    def adv_opt_string(self):
        opts = self.adv_opt_var.get().strip()
        if not opts:
            return
        self._adv_set(f"Options {opts}", f"driver options '{opts}'",
                      lambda sdr: xcall("rtlsdr_set_opt_string", sdr.dev_p, opts.encode(), 1))

    def adv_opt_help(self):
        if "rtlsdr_get_opt_help" not in EXT:
            self.log("WARN", "This librtlsdr build has no option help")
            return
        win = tk.Toplevel(self)
        win.title("librtlsdr driver options")
        t = self._mono_text(win, 32, 100)
        t.pack(fill="both", expand=True)
        t.insert("end", (_lib.rtlsdr_get_opt_help(1) or b"").decode(errors="replace").expandtabs(4))
        t.configure(state="disabled")

    # ------------------------------------------------------------- helpers
    def log(self, level, msg):
        """Thread-safe logging."""
        self.log_q.put((level, msg))

    def _flush_log(self):
        while True:
            try:
                level, msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            prefix = f"[{level}] " if level in ("PASS", "FAIL", "WARN") else ""
            self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} {prefix}{msg}\n", level)
            self.log_text.see("end")

    def dev_index(self):
        i = self.dev_combo.current()
        return self.devices[i]["index"] if i >= 0 and self.devices else 0

    def cur_rate(self):
        return float(self.rate_var.get().split()[0]) * 1e6

    def cur_freq(self):
        return float(self.freq_var.get().replace(",", ".")) * 1e6

    def cur_gain(self):
        if self.agc_var.get() or not self.gains:
            return "auto"
        return self.gains[int(self.gain_scale.get())]

    def busy(self):
        return self.selftest_thread is not None and self.selftest_thread.is_alive()

    # ------------------------------------------------------------- device
    def refresh_devices(self):
        if RTLSDR_IMPORT_ERROR:
            self.log("FAIL", f"pyrtlsdr / librtlsdr not available: {RTLSDR_IMPORT_ERROR}")
            self.log("INFO", 'Install with:  pip install "pyrtlsdr[lib]"')
            self.start_btn.state(["disabled"])
            self.test_btn.state(["disabled"])
            return
        try:
            self.devices = list_devices()
        except Exception as exc:
            self.log("FAIL", f"Enumeration failed: {exc}")
            self.devices = []
        labels = [f"#{d['index']} {d['manufacturer']} {d['product']} (SN {d['serial']})" for d in self.devices]
        self.dev_combo["values"] = labels
        if labels:
            self.dev_combo.current(0)
            self.log("INFO", f"Found {len(labels)} device(s)")
            if not self.stream:
                self.probe_device()
        else:
            self.dev_var.set("")
            self.dev_info.config(text="No device found.\nCheck USB cable and WinUSB driver (Zadig).")
            self.log("FAIL", "No RTL-SDR device found")

    def probe_device(self):
        """Open briefly to read tuner type and gain table."""
        if self.stream or self.busy():
            return
        try:
            sdr = RtlSdr(device_index=self.dev_index())
        except Exception as exc:
            self.dev_info.config(text=f"Cannot open: {exc}\n(Is another SDR program running?)")
            self.log("FAIL", f"Cannot open device: {exc}")
            return
        try:
            tuner = sdr.get_tuner_type()
            self.gains = list(sdr.valid_gains_db)
        finally:
            sdr.close()
        self.tuner_type = tuner
        self._adv_update_caps()
        d = self.devices[self.dev_combo.current()]
        lo, hi = TUNER_RANGES.get(tuner, ("?", "?"))
        self.dev_info.config(text=(f"{d['manufacturer']} {d['product']}\nSerial: {d['serial']}\n"
                                   f"Chip: {d['name']}\nTuner: {TUNER_NAMES.get(tuner, tuner)}\n"
                                   f"Range: ~{lo}-{hi} MHz\nGain steps: {len(self.gains)}"))
        self.gain_scale.configure(to=max(0, len(self.gains) - 1))
        if self.gains:
            # default to a mid-high gain (~30 dB)
            idx = min(range(len(self.gains)), key=lambda i: abs(self.gains[i] - 29.7))
            self.gain_scale.set(idx)
        self.on_gain()
        self.log("PASS", f"Opened device: tuner {TUNER_NAMES.get(tuner, tuner)}, {len(self.gains)} gain steps")

    # ------------------------------------------------------------- stream
    def toggle_stream(self):
        if self.stream:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if self.busy() or not self.devices:
            return
        try:
            fc = self.cur_freq()
        except ValueError:
            messagebox.showerror("Frequency", "Invalid frequency")
            return
        try:
            self.stream = SdrStream(self.dev_index(), self.cur_rate(), fc, self.cur_gain(),
                                    self.ppm_var.get(), int(self.fft_var.get()), self.bias_var.get(),
                                    dithering=self.adv_dither_var.get())
        except Exception as exc:
            self.stream = None
            messagebox.showerror("Open failed", f"Could not start the device:\n{exc}\n\n"
                                                "Close other SDR programs and check the WinUSB driver.")
            return
        self.stream.envelope_bin_s = float(self.envelope_bin_ms_var.get()) / 1000.0
        self.avg = self.peak = self.wf = None
        self.ylim = None
        self.last_seq = -1
        self.rate_hist.clear()
        self.reset_burst_history()
        self._adv_apply_all()  # queued: runs on the reader thread before the first read
        self.stream.start()
        self.start_btn.config(text="■ Stop")
        self.test_btn.state(["disabled"])
        self.log("INFO", f"Streaming @ {self.cur_rate()/1e6:.3f} MS/s, {fc/1e6:.3f} MHz, gain {self.cur_gain()}")
        if self.audio_var.get():
            self.on_audio()

    def stop_stream(self):
        if not self.stream:
            return
        ok = self.stream.stop()
        self.stream = None
        if not ok:
            self.log("WARN", "USB reader did not stop cleanly; replug the dongle if it misbehaves")
        self.start_btn.config(text="▶ Start")
        self.test_btn.state(["!disabled"])
        self.status_vars["State"].set("Stopped")
        self.log("INFO", "Stopped")

    def _ctrl(self, what, fn, *args):
        if self.stream:
            self.stream.control(what, fn, *args)  # errors come back via stream.error

    # ------------------------------------------------------------- callbacks
    def apply_freq(self):
        try:
            fc = self.cur_freq()
        except ValueError:
            self.log("WARN", "Invalid frequency")
            return
        self.freq_var.set(f"{fc/1e6:.3f}")
        self.avg = self.peak = None
        if self.stream:
            self._ctrl("Tuning", self.stream.tune, fc)

    def step_freq(self, step_mhz):
        try:
            self.freq_var.set(f"{self.cur_freq()/1e6 + step_mhz:.3f}")
        except ValueError:
            return
        self.apply_freq()

    def on_preset(self, _e=None):
        name = self.preset_var.get()
        for label, f in PRESETS:
            if label == name:
                self.freq_var.set(f"{f:.3f}")
                self.apply_freq()

    def on_click(self, ev):
        if ev.inaxes in (self.ax_spec, self.ax_wf) and ev.xdata is not None and ev.button == 1:
            if self.canvas.toolbar is None or not getattr(self.canvas.toolbar, "mode", ""):
                self.freq_var.set(f"{ev.xdata:.3f}")
                self.apply_freq()

    def on_rate(self):
        if self.stream:  # sample rate change requires a restart
            self.stop_stream()
            self.start_stream()

    def on_fft(self):
        if self.stream:
            self.stream.nfft = int(self.fft_var.get())
            self.avg = self.peak = self.wf = None

    def on_ppm(self):
        try:
            ppm = int(self.ppm_var.get())
        except (tk.TclError, ValueError):
            return
        if self.stream:
            self._ctrl("PPM", set_ppm, self.stream.sdr, ppm)

    def on_gain(self):
        g = self.cur_gain()
        self.gain_label.config(text=f"Gain: {'AGC' if g == 'auto' else f'{g:.1f} dB'}")
        if self.agc_var.get():
            self.gain_scale.configure(state="disabled")
        else:
            self.gain_scale.configure(state="normal")
        if self.stream:
            self._ctrl("Gain", apply_gain, self.stream.sdr, g)
            # The sidebar gain now owns the tuner gain and the RTL AGC, so stop
            # re-applying conflicting values from the Radio parameters tab.
            self._adv_forget("Tuner gain", "IF mode", "Digital AGC")
            self.adv_gainmode_var.set("auto" if g == "auto" else "manual")
            self.adv_dagc_var.set(g == "auto")

    def on_bias(self):
        if self.bias_var.get():
            if not messagebox.askyesno(
                "Bias-tee",
                "This puts ~4.5 V DC on the antenna connector to power an LNA / active antenna.\n\n"
                "Only enable it if your antenna or LNA expects it. Continue?"):
                self.bias_var.set(False)
                return
        if self.stream:
            self._ctrl("Bias-tee", self.stream.sdr.set_bias_tee, self.bias_var.get())
            self.log("INFO", f"Bias-tee {'ON' if self.bias_var.get() else 'OFF'}")

    def audio_mode(self):
        return "am" if self.audio_mode_var.get() == "AM airband voice" else "wfm"

    def on_audio_mode(self, _event=None):
        """Apply a newly selected demodulator immediately when audio is live."""
        if self.audio_var.get():
            self.on_audio()

    def on_audio(self):
        mode = self.audio_mode()
        name = "AM airband audio" if mode == "am" else "FM audio"
        if not self.stream:
            if self.audio_var.get():
                self.log("INFO", f"{name} will start with the stream")
            return
        if self.audio_var.get():
            required_rate = AUDIO_RATE if mode == "am" else IF_RATE
            if int(self.stream.fs) % required_rate:
                unit = "48 kHz (for AM)" if mode == "am" else "240 kHz (for FM)"
                self.log("WARN", f"{name} needs a sample rate that is a multiple of {unit} "
                                 "(0.96/1.2/1.44/1.92/2.4/2.88 MS/s)")
                self.audio_var.set(False)
                return
            try:
                self.stream.set_audio(True, self.vol_scale.get() / 100, mode)
                if mode == "am":
                    self.log("INFO", "AM airband audio ON - tune exactly onto an active voice channel")
                else:
                    self.log("INFO", "FM audio ON - tune exactly onto a station (click its peak)")
            except Exception as exc:
                self.audio_var.set(False)
                self.log("FAIL", f"{name} output failed: {exc}")
        else:
            self.stream.set_audio(False)

    def on_volume(self):
        if self.stream and self.stream.audio:
            self.stream.audio.volume = self.vol_scale.get() / 100

    # ------------------------------------------------------------- self-test
    def run_selftest(self):
        if self.busy() or not self.devices:
            return
        if self.stream:
            self.stop_stream()
        self.start_btn.state(["disabled"])
        self.test_btn.state(["disabled"])
        test = SelfTest(self.dev_index(), self.log)
        self.selftest_thread = threading.Thread(target=test.run, daemon=True)
        self.selftest_thread.start()
        self.status_vars["State"].set("Self-test running...")

    # ------------------------------------------------------------- poll loop
    def _poll(self):
        self._flush_log()
        try:
            if self.selftest_thread and not self.selftest_thread.is_alive():
                self.selftest_thread = None
                self.start_btn.state(["!disabled"])
                self.test_btn.state(["!disabled"])
                self.status_vars["State"].set("Stopped")
            if self.stream:
                if self.stream.error:
                    err = self.stream.error
                    self.stream.error = None
                    self.log("FAIL", f"Stream error: {err}")
                if not self.stream.reader.is_alive():
                    self.log("FAIL", "USB reader stopped (device unplugged or driver error)")
                    self.stop_stream()
                if self.stream:
                    self._update_plot()
                    self._update_status()
            self._adv_poll()
        except Exception as exc:
            self.log("FAIL", f"GUI update error: {exc!r}")
        self.after(60, self._poll)

    def _update_plot(self):
        s = self.stream
        self._update_burst_history(s)
        with s.psd_lock:
            psd, seq = s.psd, s.psd_seq
        if psd is None or seq == self.last_seq:
            return
        self.last_seq = seq
        n = len(psd)
        fc = s.fc
        freqs = (fc + (np.arange(n) - n / 2) * s.fs / n) / 1e6

        alpha = 1.0 / self.avg_scale.get()
        if self.avg is None or len(self.avg) != n:
            self.avg = psd.copy()
        else:
            self.avg = alpha * psd + (1 - alpha) * self.avg
        self.line.set_data(freqs, self.avg)
        if self.peakhold_var.get():
            self.peak = self.avg.copy() if self.peak is None or len(self.peak) != n else np.maximum(self.peak, self.avg)
            self.peak_line.set_data(freqs, self.peak)
        else:
            self.peak_line.set_data([], [])
        self.center_line.set_xdata([fc / 1e6, fc / 1e6])

        # auto-scale with hysteresis
        floor = float(np.percentile(self.avg, 20))
        top = float(self.avg.max())
        target = (floor - 10, max(top + 5, floor + 40))
        self.ylim = target if self.ylim is None else tuple(0.9 * a + 0.1 * b for a, b in zip(self.ylim, target))
        self.ax_spec.set_ylim(*self.ylim)
        self.ax_spec.set_xlim(freqs[0], freqs[-1])

        # waterfall row (max-pooled to WF_MAX_COLS)
        row = psd
        if n > WF_MAX_COLS:
            row = psd.reshape(WF_MAX_COLS, n // WF_MAX_COLS).max(axis=1)
        if self.wf is None or self.wf.shape[1] != len(row):
            self.wf = np.full((WF_ROWS, len(row)), -200.0)  # renders as the darkest colour
        self.wf[1:] = self.wf[:-1]
        self.wf[0] = row
        self.wf_img.set_data(self.wf)
        self.wf_img.set_extent((freqs[0], freqs[-1], WF_ROWS, 0))
        self.wf_img.set_clim(self.ylim[0] + 5, self.ylim[1])

        # peak marker for status
        k = int(np.argmax(self.avg))
        self._peak_info = (freqs[k], self.avg[k])
        self.canvas.draw_idle()

    def _update_burst_history(self, stream):
        """Append received power bins and render the user-selected recent span."""
        if not self.burst_display_var.get():
            return

        received = False
        while True:
            try:
                envelope, duration = stream.envelope_q.get_nowait()
            except queue.Empty:
                break
            dt = duration / len(envelope)
            self.burst_history.append((self.burst_history_end, envelope, dt))
            self.burst_history_end += duration
            received = True

        if not received or not self.burst_history:
            return

        span = float(self.burst_span_var.get())
        cutoff = self.burst_history_end - span
        while len(self.burst_history) > 1:
            start, values, dt = self.burst_history[0]
            if start + len(values) * dt >= cutoff:
                break
            self.burst_history.popleft()

        times, values = [], []
        for start, envelope, dt in self.burst_history:
            t = start + np.arange(len(envelope)) * dt
            keep = t >= cutoff
            if np.any(keep):
                times.append(t[keep] - self.burst_history_end)
                values.append(envelope[keep])
        if not values:
            return
        times = np.concatenate(times)
        values = np.concatenate(values)
        # Avoid spending UI time drawing more points than the screen can show.
        stride = max(1, math.ceil(len(values) / 8000))
        self.burst_line.set_data(times[::stride], values[::stride])
        floor = float(np.percentile(values, 10))
        top = float(np.percentile(values, 99))
        self.ax_burst.set_xlim(-min(span, self.burst_history_end), 0)
        self.ax_burst.set_ylim(floor - 8, max(top + 4, floor + 25))

    def _update_status(self):
        now = time.perf_counter()
        if self.rate_hist and now - self.rate_hist[-1][0] < 0.25:
            return
        s = self.stream
        self.rate_hist.append((now, s.total_samples))
        sv = self.status_vars
        sv["State"].set("Streaming")
        if len(self.rate_hist) >= 2:
            (t0, n0), (t1, n1) = self.rate_hist[0], self.rate_hist[-1]
            rate = (n1 - n0) / max(t1 - t0, 1e-6)
            sv["Throughput"].set(f"{rate/1e6:.3f} MS/s ({100*rate/s.fs:.1f}%)")
        sv["Dropped"].set(f"{s.dropped_blocks} blocks (DSP overload)")
        st = s.stats
        if st:
            sv["Level"].set(f"{st['rms_db']:.1f} dBFS RMS")
            sv["Clipping"].set(f"{st['clip']:.2f}%" + ("  lower gain!" if st["clip"] > 1 else ""))
            sv["DC / IQ"].set(f"DC {st['dc']:.4f}, IQ {st['iq_db']:+.2f} dB")
        if getattr(self, "_peak_info", None):
            f, p = self._peak_info
            sv["Peak"].set(f"{f:.4f} MHz @ {p:.1f} dBFS")
        a = s.audio
        if a:
            mode = "AM" if s.audio_mode == "am" else "FM"
            sv["Audio"].set(f"{mode} on, {a.queued/AUDIO_RATE*1000:.0f} ms buffered, {a.underruns} underruns")
        else:
            sv["Audio"].set("off")

    # ------------------------------------------------------------- close
    def on_close(self):
        if self.stream:
            self.stop_stream()
        self.destroy()


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
