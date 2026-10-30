"""Sound cues for a maximal contraction (MVIC): short beeps, made on the fly (no sound files, no extra libraries).

The cues follow the markers the program already writes. "Registered" means the MVIC start marker is written: by the automatic
detection from the force (the ramp was seen) or by the S key. From then on the 1-5-1 tempo of the protocol is followed
(ramp 1 s, hold, relax 1 s; the whole contraction is the 'MVIC length' setting, 7 s by default):

  Record pressed         one mid beep                    "record"  the recording has started
  ramp registered        two rising beeps                "ramp"    the ramp was seen (or S was pressed): this is the start marker
  ramp over (+1 s)       one longer high beep            "hold"    the ramp should be done: hold the force constant
  during the hold        a short tick every second       "push"    motivation: keep pushing (0 = off)
  relax (length - 1 s)   two falling beeps               "relax"   let go, over one second
  end marker written     one low beep                    "end"     the contraction is over
  rest over              three short beeps               "rest"    the rest of the SOP is over: ready for the next trial (0.27.0;
                                                                   its own switch, 'Beep when the rest is over', not 'Sound cues')

The times after the ramp are counted from when the force began to rise (the detector's estimate, about 0.1 to 0.2 s before the
start marker), or from the S key press. They are cues, not measurements: nothing here is written into a recording except the
settings used (meta.json, `audio_cues`), because a participant who hears them may push differently.

Sound is played with the Windows `winsound` module on its own thread, so a slow camera cannot make a beep late. Python's winsound
cannot play an in-memory sound asynchronously (it raises RuntimeError), so each cue is played to its end on that thread; the plan
keeps cues from overlapping. A sound that cannot be played is never allowed to disturb a recording, but it is not hidden either:
`CuePlayer.last_error` keeps the reason and the window and meta.json report it. It is silent on a machine without Windows sound,
with the environment variable LIMBTRACK_SILENT set, and in the automatic tests (offscreen Qt).
"""
import heapq
import io
import math
import os
import struct
import threading
import time
import wave
from collections import deque

RAMP_S = 1.0                 # protocol tempo: ramp up for 1 s ...
RELAX_S = 1.0                # ... hold, relax over 1 s
PUSH_INTERVAL_S = 1.0        # one motivating tick this often during the hold (0 = none)
PUSH_CLEARANCE_S = 0.25      # no tick this close to the relax cue
LATE_S = 0.25                # a cue more than this late (the force onset estimate lies in the past) is dropped
GAP_S = 0.15                 # kept free after a cue before the next may start: Windows' sound call returns about 0.1 s after the cue ends
WARM_S = 20.0                # the sound device is warmed up with a silent buffer unless a cue was played this recently
SILENCE_MS = 40

SAMPLE_RATE = 22050
VOLUME = 0.55                # of full scale
FADE_S = 0.008               # fade in and out of every tone (no clicks)
TONE_GAP_S = 0.015           # silence between the tones of one cue

# name -> tones (frequency in Hz, length in ms)
CUES = {
    "record": [(880, 150)],
    "ramp": [(660, 90), (990, 110)],
    "hold": [(1100, 280)],
    "push": [(1320, 45)],
    "relax": [(990, 120), (660, 170)],
    "end": [(440, 360)],
    "rest": [(784, 110), (784, 110), (1047, 220)],
}


def duration_s(name):
    tones = CUES[name]
    return sum(ms for _, ms in tones) / 1000.0 + TONE_GAP_S * (len(tones) - 1)


_WAV = {}


def wav_bytes(name):
    """The cue as a 16-bit mono WAV in memory. ("silence" is a short silent buffer used to wake the sound device.)"""
    if name not in _WAV:
        frames = bytearray()
        if name == "silence":
            frames += b"\x00\x00" * int(SAMPLE_RATE * SILENCE_MS / 1000.0)
        for k, (freq, ms) in enumerate(CUES.get(name, [])):
            n = int(SAMPLE_RATE * ms / 1000.0)
            fade = max(int(SAMPLE_RATE * FADE_S), 1)
            for i in range(n):
                env = min(1.0, i / fade, (n - 1 - i) / fade)
                frames += struct.pack("<h", int(32767 * VOLUME * env * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)))
            if k < len(CUES[name]) - 1:
                frames += b"\x00\x00" * int(SAMPLE_RATE * TONE_GAP_S)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(frames))
        _WAV[name] = buf.getvalue()
    return _WAV[name]


def silent():
    """No sound on a machine without a screen (the automatic tests) or when asked."""
    return bool(os.environ.get("LIMBTRACK_SILENT")) or os.environ.get("QT_QPA_PLATFORM") == "offscreen"


def _play_with_windows(name):
    if silent():
        return
    try:
        import winsound
    except ImportError:
        return
    # NOT SND_ASYNC: winsound raises RuntimeError for an in-memory sound played asynchronously. This runs on the cue thread, which
    # simply waits for the cue (at most 0.4 s) to finish.
    winsound.PlaySound(wav_bytes(name), winsound.SND_MEMORY)


def plan(ref, now, length_s, ramp_s=RAMP_S, relax_s=RELAX_S, push_every_s=PUSH_INTERVAL_S):
    """The cues that follow the registration of the ramp, as (time, name) in order. All times are on one clock (seconds).

    `ref`: when the ramp began (the force-onset estimate; or `now` when the start was marked by hand). `now`: when it was registered.
    Cues that fall more than LATE_S in the past are dropped; no cue starts before the one before it has finished."""
    items = [(now, "ramp")]
    hold_at, relax_at = ref + ramp_s, ref + length_s - relax_s
    if relax_at > hold_at:
        items.append((hold_at, "hold"))
        if push_every_s and push_every_s > 0:
            t = hold_at + push_every_s
            while t < relax_at - PUSH_CLEARANCE_S:
                items.append((t, "push"))
                t += push_every_s
    if relax_at > now - LATE_S and length_s > ramp_s:
        items.append((relax_at, "relax"))
    out, prev_t, prev_name = [], None, None
    for t, name in items:
        if t < now - LATE_S:
            continue
        if prev_t is not None:
            t = max(t, prev_t + duration_s(prev_name) + GAP_S)
        out.append((t, name))
        prev_t, prev_name = t, name
    return out


class CuePlayer:
    """Plays cues at the times it is given, on its own thread. `backend(name)` makes the sound (default: Windows `winsound`)."""

    def __init__(self, backend=None, clock=time.perf_counter):
        self._backend = backend or _play_with_windows
        self._clock = clock
        self._heap = []
        self._seq = 0
        self._cv = threading.Condition()
        self._thread = None
        self._closed = False
        self.played = deque(maxlen=200)          # (name, time it was due, time it was played): for tests and diagnostics
        self.last_error = None                   # why the latest cue could not be played ("RuntimeError: ..."), or None
        self.error_count = 0
        self._last_sound = None                  # when a sound (or the silent warm-up) was last played

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._closed = False
            self._thread = threading.Thread(target=self._run, daemon=True, name="audio-cues")
            self._thread.start()

    def schedule(self, items):
        """Add (time, name) cues. Times are on this player's clock (time.perf_counter by default)."""
        with self._cv:
            for t, name in items:
                heapq.heappush(self._heap, (float(t), self._seq, name))
                self._seq += 1
            self._ensure_thread()
            self._cv.notify_all()

    def play(self, name):
        """Play a cue now."""
        self.schedule([(self._clock(), name)])

    def warm_up(self):
        """Wake the sound device with a silent buffer, unless something was played in the last WARM_S seconds. The first call to
        Windows' sound system takes about 0.3 s longer than the following ones, and a beep that is late is a poor cue."""
        now = self._clock()
        if self._last_sound is None or now - self._last_sound > WARM_S:
            self._last_sound = now                   # (so two quick calls do not queue two warm-ups)
            self.schedule([(now, "silence")])

    def cancel(self):
        """Forget every cue that has not been played yet."""
        with self._cv:
            self._heap.clear()
            self._cv.notify_all()

    def pending(self):
        with self._cv:
            return [(t, n) for t, _, n in sorted(self._heap)]

    def close(self):
        with self._cv:
            self._closed = True
            self._heap.clear()
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while not self._closed:
                    if not self._heap:
                        self._cv.wait()
                        continue
                    delay = self._heap[0][0] - self._clock()
                    if delay <= 0:
                        break
                    self._cv.wait(timeout=delay)
                if self._closed:
                    return
                due, _, name = heapq.heappop(self._heap)
            started = self._clock()
            try:
                self._backend(name)
            except Exception as e:  # noqa: BLE001 - a sound problem must never reach the recording, but it is kept and reported
                self.last_error = f"{type(e).__name__}: {e}"
                self.error_count += 1
            self._last_sound = self._clock()
            if name != "silence":
                self.played.append((name, due, started))      # (due, and when the call to the sound system was made)
