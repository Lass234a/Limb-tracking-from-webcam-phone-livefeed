"""Force from a load cell through a data acquisition (DAQ) device, sampled on its own thread at its own, much higher, rate.

The camera delivers about 30 pictures per second; a maximal contraction is over in a few seconds and its peak is short, so the
force is NOT sampled once per picture. The DAQ samples at SAMPLE_RATE_HZ (1000 per second) and every sample is saved with a
time on the same clock as the video frames (time.perf_counter), so a force sample can be matched to the joint angle at that moment.

Pieces
  * ForceCalibration  : volts -> newtons (zero offset and newtons per volt). The slope comes from the load cell's regression
                        (limbtrack/loadcell.py); the zero is taken by the operator and is NOT kept between sessions.
  * ForceSource       : keeps the newest samples (ring buffer) and gives them out by time; the hardware classes only differ in how
                        they obtain samples.
  * NiDaqForce        : an NI DAQmx device (written for the USB-6003; any DAQmx device with analog inputs works).
  * SimulatedForce    : a made-up contraction every few seconds, for testing and for trying the program without hardware.

Timing: a DAQ chunk is handed to the program a little after its last sample was taken (USB and driver delay). The time of sample 0
is therefore estimated as the smallest 'arrival time minus samples/rate' seen over the last seconds (ClockAligner). The remaining
uncertainty is the smallest delay, a few milliseconds. Pressing the MVIC start marker gives a second, independent, time stamp.
"""
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .loadcell import GRAVITY, active_regression

SAMPLE_RATE_HZ = 1000.0
CHUNK_S = 0.01                    # the reader asks the device for this much at a time
KEEP_SECONDS = 600.0              # samples kept in memory (a recording is far shorter)
STANDARD_GRAVITY = GRAVITY        # m/s^2: a mass of 1 kg weighs 9.80665 N
MIN_CAL_SPAN_V = 0.005            # a known load must move the signal by at least this much for a two-point calibration
ALIGN_WINDOW_S = 20.0
CLIP_FRACTION = 0.99                # a sample this close to +-range is 'at the limit'
SMOOTH_DEFAULT_S = 0.1              # display and analysis smoothing: a moving average of this length (a multiple of 20 ms removes 50 Hz hum)
NO_FORCE_RISE_N = 60.0              # a contraction moves the force (0.1 s average) at least this far from its rest; less is 'no force rise'
                                    # (P20, no load reached the DAQ: at most 47 N; P11: at least 112 N). Used after a trial (0.19.0), and by
                                    # the 10 % check and the resting-force note, which leave trials without a rise out (0.26.0: P02 2026-10-29)
MAINS_HZ = 50.0
MAX_STEADY_SD_V = 0.03            # noise allowed in a zero or calibration reading: the spread of the signal WITHOUT its 50 Hz mains hum
                                  # (was 0.02 on the whole spread, hum included, until 0.10.0). The hum averages out of the mean exactly
                                  # (the window is a whole number of mains cycles), so it does not make the zero less accurate; random
                                  # noise averages out too: the mean of N samples is good to noise / sqrt(N) (30 mV over 1000 samples: 1 mV).
MAX_DRIFT_V = 0.01                # the first and the second half of a reading may differ by at most this: a drifting signal is what would
                                  # bias the zero (a person leaning on the rig, a settling strap), and averaging does not remove it
HUM_NOTE_V = 0.01                 # a zero taken with at least this much 50 Hz hum is reported with a note about the charger and outlet
ZERO_SECONDS = 1.0                # the zero is the mean of this much signal (a multiple of 20 ms, so the 50 Hz hum averages out exactly)
RELAXED_SECONDS = 2.0             # the relaxed level (0.22.0) is the mean of this much: a strap still settling or a leg still relaxing drifts
                                  # more than MAX_DRIFT_V over 2 s and is refused, so the reading is only taken once the tension has settled



class ForceError(RuntimeError):
    """The force source could not be opened or stopped delivering; the message says what to do."""


# ------------------------------------------------------------------------------------------- smoothing and hum
def smooth_signal(values, rate, window_s):
    """Centred moving average of `window_s` seconds (edges: the end values are repeated). 0 or one sample = unchanged.
    At 1000 samples per second a window that is a multiple of 20 ms (20, 40, 100, 200 ms ...) removes 50 Hz mains hum and its
    harmonics exactly, because it then covers a whole number of hum cycles. The saved samples are never smoothed."""
    values = np.asarray(values, dtype=np.float64)
    n = int(round(window_s * rate))
    if n <= 1 or values.size == 0:
        return values.copy()
    n = min(n, values.size)
    left = n // 2
    padded = np.pad(values, (left, n - 1 - left), mode="edge")
    c = np.cumsum(np.insert(padded, 0, 0.0))
    return (c[n:] - c[:-n]) / n


def mains_amplitude(volts, rate, mains_hz=MAINS_HZ):
    """Amplitude (volts, peak) of the mains-frequency part of a signal, by a lock-in over a whole number of mains cycles.
    NaN when the signal is shorter than 5 cycles."""
    x = np.asarray(volts, dtype=np.float64)
    cycles = int(x.size * mains_hz / rate)
    if cycles < 5:
        return math.nan
    n = int(round(cycles * rate / mains_hz))
    seg = x[:n] - x[:n].mean()
    t = np.arange(n) / rate
    return float(2.0 * abs(np.mean(seg * np.exp(-2j * np.pi * mains_hz * t))))


# ------------------------------------------------------------------------------------------- calibration
@dataclass
class ForceCalibration:
    """newtons = (volts - offset_v) * n_per_v. `n_per_v` is None until the load cell's slope (or a known load) has been used.
    `zero_set` is False for a slope that has not been zeroed yet: the offset would be a guess, so no newtons are given then."""
    offset_v: float = 0.0
    n_per_v: float = None
    method: str = "none"
    when: str = ""
    known_mass_kg: float = None
    zero_v: float = None
    load_v: float = None
    note: str = ""
    zero_set: bool = True
    regression_id: str = ""            # which load cell regression the slope came from ("" = a known weight or none)
    zero_noise_v: float = None         # spread of the signal (mains hum taken out) in the window the zero was taken from
    zero_hum_v: float = None           # 50 Hz hum amplitude in that window (it averages out of the zero)
    zero_se_v: float = None            # how well the zero is known: noise / sqrt(samples), in volts
    # 0.22.0: the second zero level. The first zero is the strap with nothing on it (0 N). The relaxed level is the signal with the participant
    # in the position and relaxed, the strap taut: the tension that is there before any push. It is kept apart from the zero: the newtons
    # stay the total force from the first zero, and "above relaxed" is a second number beside them. A new zero clears it (the set-up changed).
    relaxed_v: float = None            # volts at the relaxed level (None: not taken)
    relaxed_noise_v: float = None      # spread of the signal (mains hum taken out) in the window it was taken from
    relaxed_se_v: float = None         # how well the relaxed level is known: noise / sqrt(samples), in volts
    relaxed_when: str = ""
    relaxed_for: str = ""              # the position it was taken for (participant | test | target): the tension differs with every position

    @classmethod
    def load_cell(cls, regression=None):
        """The load cell's own slope (the active regression unless one is given), waiting for a zero. Without any regression: volts only."""
        reg = regression or active_regression()
        if reg is None:
            return cls()
        return cls(n_per_v=reg.n_per_v, method=f"load cell regression {reg.id}", regression_id=reg.id, zero_set=False)

    @property
    def has_slope(self):
        return self.n_per_v is not None and self.n_per_v != 0 and math.isfinite(self.n_per_v)

    @property
    def calibrated(self):
        return self.has_slope and self.zero_set

    def to_newton(self, volts):
        """Newtons for volts (a number or an array); NaN when not calibrated."""
        v = np.asarray(volts, dtype=np.float64)
        return (v - self.offset_v) * self.n_per_v if self.calibrated else np.full(v.shape, np.nan)

    def set_zero(self, volts, noise_v=None, hum_v=None, samples=None):
        """The signal with nothing loaded. `noise_v`, `hum_v` and `samples` describe the window it was taken from (saved with the trial)."""
        self.offset_v, self.zero_v, self.zero_set = float(volts), float(volts), True
        self.zero_noise_v = None if noise_v is None else float(noise_v)
        self.zero_hum_v = None if hum_v is None else float(hum_v)
        self.zero_se_v = float(noise_v) / math.sqrt(samples) if noise_v is not None and samples else None
        self.when = datetime.now().isoformat(timespec="seconds")
        self.method = "zero only" if not self.has_slope else self.method
        self.clear_relaxed()                                     # a relaxed level belongs to the zero it was taken after

    def set_relaxed(self, volts, noise_v=None, samples=None, for_what=""):
        """The signal with the participant relaxed in the strap (the zero must have been taken before)."""
        self.relaxed_v = float(volts)
        self.relaxed_noise_v = None if noise_v is None else float(noise_v)
        self.relaxed_se_v = float(noise_v) / math.sqrt(samples) if noise_v is not None and samples else None
        self.relaxed_when = datetime.now().isoformat(timespec="seconds")
        self.relaxed_for = for_what

    def clear_relaxed(self):
        self.relaxed_v = self.relaxed_noise_v = self.relaxed_se_v = None
        self.relaxed_when = self.relaxed_for = ""

    @property
    def relaxed_n(self):
        """The relaxed tension in newtons above the zero, or None when it is not taken or the force is not calibrated."""
        return float(self.to_newton(self.relaxed_v)) if self.relaxed_v is not None and self.calibrated else None

    @property
    def zero_se_n(self):
        """The zero's uncertainty in newtons (one standard error), or None when it is not known or there is no slope."""
        return self.zero_se_v * self.n_per_v if self.zero_se_v is not None and self.has_slope else None

    def calibrate_with_mass(self, load_volts, mass_kg):
        """Two points: the zero set before (zero_v) and a known mass hanging on, or pushing on, the sensor.
        Returns None if fine, else a plain-English reason why it was refused (nothing is changed then)."""
        if mass_kg <= 0:
            return "The known mass must be above zero."
        span = float(load_volts) - self.offset_v
        if abs(span) < MIN_CAL_SPAN_V:
            return (f"The signal moved only {span * 1000:+.1f} mV from the zero. Check the load is on the sensor and the "
                    f"amplifier is connected (at least {MIN_CAL_SPAN_V * 1000:.0f} mV are needed).")
        self.n_per_v = mass_kg * STANDARD_GRAVITY / span
        self.known_mass_kg, self.load_v = float(mass_kg), float(load_volts)
        self.method, self.regression_id = f"zero and {mass_kg:g} kg", ""
        self.when = datetime.now().isoformat(timespec="seconds")
        return None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()


# ------------------------------------------------------------------------------------------- clock
class ClockAligner:
    """Estimates when sample 0 was taken, on the perf_counter clock, from when chunks arrived (see the module note)."""

    def __init__(self, rate, window_s=ALIGN_WINDOW_S):
        self.rate, self.window_s = float(rate), window_s
        self._mins = deque()            # (arrival, arrival - n_end / rate), increasing in the second number: the sliding-window minimum
        self.t0 = None

    def update(self, arrival, n_end):
        v = arrival - n_end / self.rate
        while self._mins and self._mins[-1][1] >= v:
            self._mins.pop()
        self._mins.append((arrival, v))
        while self._mins[0][0] < arrival - self.window_s:
            self._mins.popleft()
        self.t0 = self._mins[0][1]
        return self.t0

    def times(self, first_index, count):
        """Times of samples first_index .. first_index + count - 1 (the last of a chunk is at arrival - latency)."""
        return self.t0 + (np.arange(first_index + 1, first_index + count + 1) / self.rate)


# ------------------------------------------------------------------------------------------- sources
class ForceSource:
    kind = "none"

    def __init__(self, rate=SAMPLE_RATE_HZ):
        self.rate = float(rate)
        self.calibration = ForceCalibration()
        self.error = None                         # why the reading stopped, in plain English
        self.read_errors = 0
        self._lock = threading.Lock()
        self._chunks = deque()                    # (times, volts) arrays
        self._kept = 0
        self._n_total = 0
        self._aligner = ClockAligner(self.rate)
        self._thread = None
        self._stop = threading.Event()
        self._last_arrival = None
        self.frozen = False                       # set while a trial is recorded: zero and calibration cannot change then
        self.range_v = 10.0                       # the input range is +-range_v volts
        self.smooth_s = SMOOTH_DEFAULT_S          # what the live read-out and trace show (display only; the files hold every raw sample)
        self.clipped_total = 0                    # samples at the very edge of the range (the signal may be cut off there)
        self.last_reading = None                  # what the latest zero / calibration reading measured (see _steady_mean)
        self._last_clip = None

    # --- life cycle
    def start(self):
        """Open the hardware (errors are raised here, with advice) and start reading."""
        self._open()
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="force-reader")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        try:
            self._close()
        except Exception:  # noqa: BLE001 - closing must never raise into the window
            pass

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def _open(self):
        pass

    def _close(self):
        pass

    def _read_chunk(self):
        """Hardware classes: block until the next chunk and return its voltages as a 1-D array."""
        raise NotImplementedError

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                chunk = np.asarray(self._read_chunk(), dtype=np.float64).ravel()
            except Exception as e:  # noqa: BLE001 - reported to the operator
                self.read_errors += 1
                self.error = self._explain(e)
                return
            if chunk.size:
                self._push(chunk, time.perf_counter())

    def _explain(self, e):
        return f"The force reading stopped: {e}"

    def _push(self, chunk, arrival):
        with self._lock:
            n0 = self._n_total
            self._n_total += chunk.size
            self._aligner.update(arrival, self._n_total)
            t = np.maximum.accumulate(self._aligner.times(n0, chunk.size))
            if self._chunks and t[0] < self._chunks[-1][0][-1]:         # the clock estimate moved: keep time running forward
                t = np.maximum(t, self._chunks[-1][0][-1])
            self._chunks.append((t, chunk))
            self._kept += chunk.size
            self._last_arrival = arrival
            clipped = int(np.count_nonzero(np.abs(chunk) >= CLIP_FRACTION * self.range_v))
            if clipped:
                self.clipped_total += clipped
                self._last_clip = arrival
            while self._kept - self._chunks[0][1].size > KEEP_SECONDS * self.rate:
                self._kept -= self._chunks.popleft()[1].size

    # --- reading what has come in
    def seconds_since_last_sample(self):
        return float("inf") if self._last_arrival is None else time.perf_counter() - self._last_arrival

    def clipping_now(self):
        """True if the signal touched the edge of the input range within the last second."""
        return self._last_clip is not None and time.perf_counter() - self._last_clip < 1.0

    def _tail(self, n):
        """The last n raw voltages (fewer at the start)."""
        with self._lock:
            parts, have = [], 0
            for _, v in reversed(self._chunks):
                parts.append(v)
                have += v.size
                if have >= n:
                    break
        return np.concatenate(parts[::-1])[-n:] if parts else np.empty(0)

    def latest_volts(self):
        """The newest voltage, smoothed over smooth_s for the read-out (0 = the raw newest sample)."""
        v = self._tail(max(1, int(round(self.smooth_s * self.rate))))
        return float(v.mean()) if v.size else None

    def latest_newtons(self):
        v = self.latest_volts()
        return None if v is None or not self.calibration.calibrated else float(self.calibration.to_newton(v))

    def samples_between(self, t0, t1):
        """(times, volts) of the samples with t0 <= time < t1.

        The chunks are in time order (_push keeps time running forward), so the search walks back from the newest chunk and stops
        at the first one that ended before t0. Looking at all 10 minutes of chunks instead cost 13 ms per call once the force had
        run for 10 minutes, several times per picture, and held the live picture to 4-11 per second (P11, 2026-10-15)."""
        with self._lock:
            parts = []
            for c in reversed(self._chunks):
                if c[0][-1] < t0:
                    break                  # this chunk and every older one ended before t0
                if c[0][0] < t1:
                    parts.append(c)
            parts.reverse()
        if not parts:
            return np.empty(0), np.empty(0)
        t = np.concatenate([p[0] for p in parts])
        v = np.concatenate([p[1] for p in parts])
        m = (t >= t0) & (t < t1)
        return t[m], v[m]

    def recent(self, seconds):
        """(seconds before now - negative, volts) of roughly the last `seconds`, thinned for plotting."""
        now = time.perf_counter()
        t, v = self.samples_between(now - seconds, now + 1.0)
        v = smooth_signal(v, self.rate, self.smooth_s)                 # before thinning, so the hum does not alias into the picture
        if t.size > 1500:
            step = int(math.ceil(t.size / 1500))
            t, v = t[::step], v[::step]
        return t - now, v

    def average_volts(self, seconds=2.0):
        """Mean voltage of the last `seconds` and how many samples that was (for the zero and the known-load readings)."""
        now = time.perf_counter()
        t, v = self.samples_between(now - seconds, now + 1.0)
        return (float(v.mean()), int(v.size)) if v.size else (None, 0)

    def spread_volts(self, seconds=2.0):
        now = time.perf_counter()
        _, v = self.samples_between(now - seconds, now + 1.0)
        return float(v.std()) if v.size > 1 else None

    def _steady_mean(self, seconds, hint="Keep the sensor still and the load constant while it measures."):
        """(mean volts, None) of the last `seconds`, or (None, reason) when there is too little or the signal is not steady.

        Steady means: no drift (the two halves agree within MAX_DRIFT_V) and little noise (the spread without the 50 Hz hum is within
        MAX_STEADY_SD_V). The hum is taken out because a whole number of its cycles averages out of the mean exactly. What was
        measured is kept in `last_reading` (mean, noise, hum, drift, samples)."""
        if self.frozen:
            return None, "Stop recording first: the zero and the calibration cannot change during a recording."
        if self.error:
            return None, self.error
        now = time.perf_counter()
        _, v = self.samples_between(now - seconds, now + 1.0)
        if v.size < 0.5 * seconds * self.rate or v.size < 4:
            return None, "Not enough force samples yet. Wait a moment and try again."
        hum = mains_amplitude(v, self.rate)
        hum = hum if math.isfinite(hum) else 0.0
        noise = math.sqrt(max(float(v.std()) ** 2 - hum ** 2 / 2.0, 0.0))        # a sine's variance is amplitude squared / 2
        half = v.size // 2
        drift = float(v[half:].mean() - v[:half].mean())
        self.last_reading = {"mean_v": float(v.mean()), "noise_v": noise, "hum_v": hum, "drift_v": drift, "samples": int(v.size)}
        if abs(drift) > MAX_DRIFT_V:
            return None, (f"The signal is not steady: it moved {drift * 1000:+.0f} mV during the {seconds:g} s (limit {MAX_DRIFT_V * 1000:.0f} mV). "
                          + hint)
        if noise > MAX_STEADY_SD_V:
            return None, (f"The signal is not steady enough: it wobbles {noise * 1000:.0f} mV over {seconds:g} s (limit {MAX_STEADY_SD_V * 1000:.0f} mV, "
                          f"the 50 Hz mains hum not counted). " + hint)
        return float(v.mean()), None

    def zero(self, seconds=ZERO_SECONDS):
        """Take the present signal as 'no load'. Returns None if done, else the reason it was not."""
        v, why = self._steady_mean(seconds)
        if why:
            return why
        r = self.last_reading
        self.calibration.set_zero(v, noise_v=r["noise_v"], hum_v=r["hum_v"], samples=r["samples"])
        return None

    def relax(self, for_what="", seconds=RELAXED_SECONDS):
        """Take the present signal as the 'relaxed in the strap' level (0.22.0): the participant in the position, relaxed, the strap taut,
        not pushing. Needs the zero first. Returns None if done, else the reason it was not (nothing is changed then)."""
        if self.calibration.zero_v is None:
            return "Press Zero (Z) first, with no pull on the strap. The relaxed level is measured on top of that zero."
        v, why = self._steady_mean(seconds, hint="Ask the participant to stay relaxed and still, with the strap taut, and wait a moment.")
        if why:
            return why
        r = self.last_reading
        self.calibration.set_relaxed(v, noise_v=r["noise_v"], samples=r["samples"], for_what=for_what)
        return None

    def calibrate(self, mass_kg, seconds=2.0):
        """With a known mass on the sensor (after zero): sets newtons per volt. Returns None if done, else the reason."""
        v, why = self._steady_mean(seconds)
        if why:
            return why
        return self.calibration.calibrate_with_mass(v, mass_kg)

    def info(self):
        """What to record about the source (meta.json)."""
        return {"source": self.kind, "sample_rate_hz": self.rate, "calibration": self.calibration.to_dict(),
                "calibrated": self.calibration.calibrated}


class NiDaqForce(ForceSource):
    """An NI-DAQmx device with the load cell's amplifier on one analog input."""
    kind = "NI DAQmx"
    TERMINALS = {"differential": "DIFF", "single-ended": "RSE"}

    def __init__(self, device="Dev1", channel="ai0", terminal="differential", rate=SAMPLE_RATE_HZ, range_v=10.0):
        super().__init__(rate)
        self.device, self.channel, self.terminal, self.range_v = device, channel, terminal, float(range_v)
        self._task = None

    @staticmethod
    def devices():
        """[(name, product type, [analog input names])] of the DAQ devices the driver knows; [] without the driver or package."""
        try:
            from nidaqmx.system import System
            return [(d.name, d.product_type, [c.name.split("/")[-1] for c in d.ai_physical_chans]) for d in System.local().devices]
        except Exception:  # noqa: BLE001 - no package, no driver: nothing to list
            return []

    def _open(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        except ImportError as e:
            raise ForceError("The nidaqmx Python package is not installed (pip install nidaqmx) or the NI-DAQmx driver is missing.") from e
        cfg = getattr(TerminalConfiguration, self.TERMINALS.get(self.terminal, "DIFF"))
        task = None
        try:
            task = nidaqmx.Task()
            task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.channel}", terminal_config=cfg,
                                                 min_val=-self.range_v, max_val=self.range_v)
            task.timing.cfg_samp_clk_timing(self.rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(self.rate * 10))
            task.start()
        except Exception as e:  # noqa: BLE001 - turned into advice
            if task is not None:
                try:
                    task.close()
                except Exception:  # noqa: BLE001
                    pass
            raise ForceError(self._explain_open(e)) from e
        self._task = task

    def _explain_open(self, e):
        text = str(e)
        if "reserved" in text.lower() or "-50103" in text:
            return (f"{self.device} is in use by another program (MATLAB, NI MAX test panel, LabVIEW or another recording). "
                    f"Close it, or release the device there, then connect again.")
        if "-200220" in text or "not found" in text.lower() or "device identifier is invalid" in text.lower():
            return f"No DAQ device called '{self.device}'. Check the name in NI MAX and that the USB cable is connected."
        return f"Could not open {self.device}/{self.channel}: {text.splitlines()[0] if text else type(e).__name__}"

    def _read_chunk(self):
        n = max(int(self.rate * CHUNK_S), 1)
        return self._task.read(number_of_samples_per_channel=n, timeout=2.0)

    def _explain(self, e):
        text = str(e)
        if "overflow" in text.lower() or "-200279" in text:
            return "The program could not keep up with the DAQ and lost samples. Close other heavy programs and try again."
        if "timed out" in text.lower() or "-200284" in text:
            return "The DAQ stopped sending samples (cable unplugged or device reset)."
        return f"The force reading stopped: {text.splitlines()[0] if text else type(e).__name__}"

    def _close(self):
        task, self._task = self._task, None
        if task is not None:
            try:
                task.stop()
            finally:
                task.close()

    def info(self):
        d = super().info()
        d.update({"device": self.device, "channel": self.channel, "terminal": self.terminal, "range_v": self.range_v})
        return d


class SimulatedForce(ForceSource):
    """A made-up maximal contraction every `period_s` seconds (ramp, hold, release) plus noise: for tests and for trying the program."""
    kind = "simulated"

    def __init__(self, rate=SAMPLE_RATE_HZ, peak_n=400.0, period_s=12.0, rest_s=3.0, hold_s=3.0, ramp_s=1.5, noise_v=0.002, offset_v=0.0123,
                 n_per_v=100.0, clock=time.perf_counter, seed=0):
        super().__init__(rate)
        self.peak_n, self.period_s, self.rest_s, self.hold_s, self.ramp_s = peak_n, period_s, rest_s, hold_s, ramp_s
        self.noise_v, self._true_offset, self._true_n_per_v = noise_v, offset_v, n_per_v
        self.calibration = ForceCalibration(offset_v=offset_v, n_per_v=n_per_v, method="simulated", when="")
        self._clock, self._rng = clock, np.random.RandomState(seed)
        self._t_start = None
        self._next = 0

    def newtons_at(self, t):
        """The made-up true force t seconds after the start (an array or a number): rest, ramp up, hold, release."""
        t = np.asarray(t, dtype=np.float64) % self.period_s - self.rest_s
        up = np.clip(t / self.ramp_s, 0, 1)
        down = np.clip(1.0 - (t - self.ramp_s - self.hold_s) / 1.0, 0, 1)
        shape = np.where(t < 0, 0.0, np.where(t < self.ramp_s, up, np.where(t < self.ramp_s + self.hold_s, 1.0, down)))
        return self.peak_n * shape

    def _open(self):
        self._t_start = self._clock()
        self._next = 0

    def _read_chunk(self):
        n = max(int(self.rate * CHUNK_S), 1)
        due = self._t_start + (self._next + n) / self.rate
        wait = due - self._clock()
        if wait > 0:
            time.sleep(wait)
        idx = np.arange(self._next, self._next + n)
        self._next += n
        n_true = self.newtons_at(idx / self.rate)
        volts = n_true / self._true_n_per_v + self._true_offset + self._rng.normal(0.0, self.noise_v, n)
        return np.clip(volts, -self.range_v, self.range_v)          # like the real input: it cannot read beyond its range


def make_source(kind, **kw):
    """'ni' / 'simulated' -> a source (not started)."""
    if kind == "ni":
        return NiDaqForce(**kw)
    if kind == "simulated":
        return SimulatedForce(**{k: v for k, v in kw.items() if k in ("rate", "peak_n", "period_s", "rest_s", "ramp_s", "hold_s")})
    raise ValueError(f"unknown force source: {kind}")
