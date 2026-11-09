#!/usr/bin/env python3
"""
rtlsdr_tester.py - GUI test bench for NooElec NESDR / RTL2832U dongles.

What it does
  * Detects the dongle and shows its USB strings, tuner chip and gain table
  * Live spectrum + waterfall (click on either to tune there)
  * Controls: frequency, presets, sample rate, gain/AGC, PPM, FFT size, bias-tee
  * Live stream health: real throughput, dropped blocks, ADC level, clipping,
    DC offset, I/Q balance, strongest peak
  * Wideband FM receiver with audio out, to check reception by listening
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

    def __init__(self, index, fs, fc, gain, ppm, nfft, bias_tee=False):
        self.fs = fs
        self.fc = fc
        self.nfft = nfft
        self.cmd_q = queue.Queue()
        self.sdr = RtlSdr(device_index=index)
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
        self.stats = {}
        self.demod = None
        self.audio = None

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

    def set_audio(self, enabled, volume=0.5):
        if enabled:
            self.demod = WfmDemod(self.fs)
            self.audio = AudioSink()
            self.audio.volume = volume
        else:
            self.demod = None
            a, self.audio = self.audio, None
            if a:
                a.close()

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
                # (a retune takes ~75 ms, so slider drags would pile up)
                pending = {}
                while not self.cmd_q.empty():
                    what, fn, args = self.cmd_q.get_nowait()
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
        if nframes:
            win = np.hanning(nfft)
            frames = iq[: nframes * nfft].reshape(nframes, nfft) * win
            spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
            p = np.mean(np.abs(spec) ** 2, axis=0) / (win.sum() ** 2)
            psd = lin2db(p)
            with self.psd_lock:
                self.psd = psd
                self.psd_seq += 1
        # ---- raw sample health ----
        re, im = iq.real, iq.imag
        sre, sim = re.std(), im.std()
        self.stats = {
            "rms_db": lin2db(np.mean(re * re + im * im)),
            "clip": 100.0 * np.mean((np.abs(re) > 0.995) | (np.abs(im) > 0.995)),
            "dc": abs(iq.mean()),
            "iq_db": 20 * math.log10(max(sre, 1e-9) / max(sim, 1e-9)),
        }
        # ---- FM audio ----
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
        self.geometry("1400x900")
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

        self._build_ui()
        self.after(100, self.refresh_devices)
        self.after(50, self._poll)

    # ------------------------------------------------------------- UI build
    def _build_ui(self):
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        side = ttk.Frame(root)
        side.pack(side="left", fill="y", padx=(0, 6))
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

        # audio
        f = ttk.LabelFrame(side, text="FM audio (wideband FM)", padding=6)
        f.pack(fill="x", pady=3)
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

        # run
        f = ttk.Frame(side)
        f.pack(fill="x", pady=6)
        self.start_btn = ttk.Button(f, text="▶ Start", command=self.toggle_stream)
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.test_btn = ttk.Button(f, text="Run self-test", command=self.run_selftest)
        self.test_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

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
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="#1b1d22")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.08)
        self.ax_spec = self.fig.add_subplot(gs[0])
        self.ax_wf = self.fig.add_subplot(gs[1], sharex=self.ax_spec)
        for ax in (self.ax_spec, self.ax_wf):
            ax.set_facecolor("#111317")
            ax.tick_params(colors="#c8ccd4", labelsize=8)
            for s in ax.spines.values():
                s.set_color("#444a55")
        self.ax_spec.grid(True, color="#2c313a", lw=0.6)
        self.ax_spec.set_ylabel("dBFS", color="#c8ccd4")
        self.ax_wf.set_xlabel("Frequency (MHz)  -  click to tune", color="#c8ccd4")
        self.ax_wf.set_yticks([])
        self.line, = self.ax_spec.plot([], [], color="#4fc3f7", lw=0.9)
        self.peak_line, = self.ax_spec.plot([], [], color="#ffb74d", lw=0.7, alpha=0.8)
        self.center_line = self.ax_spec.axvline(0, color="#ef5350", lw=0.6, alpha=0.6)
        self.wf_img = self.ax_wf.imshow(np.zeros((WF_ROWS, 2)), aspect="auto", cmap="turbo",
                                        origin="upper", interpolation="nearest")
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.07)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
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
                                    self.ppm_var.get(), int(self.fft_var.get()), self.bias_var.get())
        except Exception as exc:
            self.stream = None
            messagebox.showerror("Open failed", f"Could not start the device:\n{exc}\n\n"
                                                "Close other SDR programs and check the WinUSB driver.")
            return
        self.avg = self.peak = self.wf = None
        self.ylim = None
        self.last_seq = -1
        self.rate_hist.clear()
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

    def on_audio(self):
        if not self.stream:
            if self.audio_var.get():
                self.log("INFO", "FM audio will start with the stream")
            return
        if self.audio_var.get():
            if int(self.stream.fs) % IF_RATE:
                self.log("WARN", "FM audio needs a sample rate that is a multiple of 240 kHz "
                                 "(0.96/1.2/1.44/1.92/2.4/2.88 MS/s)")
                self.audio_var.set(False)
                return
            try:
                self.stream.set_audio(True, self.vol_scale.get() / 100)
                self.log("INFO", "FM audio ON - tune exactly onto a station (click its peak)")
            except Exception as exc:
                self.audio_var.set(False)
                self.log("FAIL", f"Audio output failed: {exc}")
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
        except Exception as exc:
            self.log("FAIL", f"GUI update error: {exc!r}")
        self.after(60, self._poll)

    def _update_plot(self):
        s = self.stream
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
        sv["Audio"].set(f"on, {a.queued/AUDIO_RATE*1000:.0f} ms buffered, {a.underruns} underruns" if a else "off")

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
