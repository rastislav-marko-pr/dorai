#!/usr/bin/env python3
"""
voice_mod / voice.py  —  dorai real-time mic-array front-end.

A single ROS 2 node that runs the full capture-to-enhancement path:

    mic array (N mics, USB/other)
        -> per-mic clock-drift-corrected async resample to 16 kHz
        -> dorai_beamformer.ort multichannel beamformer (continuous, real-time)
        -> clean single-channel speech
        -> published on /dorai_clean_audio in N-second frames (default 10 s)

Enhancement runs continuously on a 3200/1600 sliding window with no added
algorithmic delay; only the publish to the topic is batched into
`publish_interval`-second frames so stt_mod receives larger, cheaper chunks.

Design properties:
  * Audio callbacks only append to bounded, per-channel locked ring buffers,
    keeping the PortAudio thread responsive.
  * A single dedicated worker thread does resample + inference + publish, so
    the ROS executor is never blocked by ONNX.
  * Per-mic asynchronous sample-rate conversion with a closed-loop ratio trim
    keeps the independent USB ADC clocks locked to one 16 kHz master timeline.
  * All buffers are bounded (drop-oldest); a stalled mic degrades gracefully to
    zeros instead of blocking the array.
"""

import os
import math
import time
import queue
import threading

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

import onnxruntime as ort

# Quiet ORT's C++ logger. On a headless Pi (no GPU) the provider bring-up
# prints harmless "GetGpuDevices ... Failed to open .../device/vendor" warnings
# at startup; raise the severity floor to errors so they don't clutter the log.
# 0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL.
try:
    ort.set_default_logger_severity(3)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

import sounddevice as sd


# ---------------------------------------------------------------------------
# Constants — the topic contract (must match stt_mod) and DSP framing.
# ---------------------------------------------------------------------------
OUTPUT_RATE = 16000               # master timeline rate (Hz)
BLOCK_MS = 100                    # worker tick period (ms)
BLOCK_S = BLOCK_MS / 1000.0
L_OUT = OUTPUT_RATE * BLOCK_MS // 1000          # output samples/block/ch (1600)

# dorai_beamformer.ort is a whole-signal model (enhanced once per publish frame, not over
# a sliding window). WINDOW_SIZE is only the dummy length used to warm the
# session up before real frames arrive.
WINDOW_SIZE = 3200

# Asynchronous-resampler drift control.
TARGET_BLOCKS = 2                 # steady-state out-FIFO depth (blocks)
TARGET_OUT = TARGET_BLOCKS * L_OUT
RATIO_TRIM_KP = 0.05              # proportional gain of the drift loop
RATIO_TRIM_MAX = 0.02            # clamp |ratio correction| to +/-2 %

# Buffer bounds (overflow protection). Normal FIFO depth sits at TARGET_OUT;
# this ceiling is only a safety net that absorbs bursts of stalled worker
# ticks before dropping oldest audio to keep latency bounded.
MAX_IN_SECONDS = 2.0              # max input backlog per mic (s)
MAX_OUT_BLOCKS = 16              # max resampled FIFO depth (blocks, ~1.6 s)

DIAG_PERIOD_S = 5.0


# ---------------------------------------------------------------------------
# Drift-correcting asynchronous resampler to 16 kHz.
# ---------------------------------------------------------------------------
class Resampler:
    """Variable-ratio resampler (libsamplerate) with a scipy polyphase fallback.

    The variable-ratio path lets the drift loop nudge the conversion ratio a
    fraction of a percent per tick so the resampler — not a growing buffer —
    absorbs the difference between a USB ADC's true clock and its nominal rate.
    The scipy fallback is fixed-ratio (no per-tick trim) but still produces an
    exact-rate stream good enough when libsamplerate is unavailable.
    """

    def __init__(self, input_rate, output_rate=OUTPUT_RATE):
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.base_ratio = output_rate / float(input_rate)
        self.backend = "unknown"
        self._r = None
        self.variable = False
        self.passthrough = False

        # One clock, one rate -> no conversion at all. A multichannel array
        # (ReSpeaker) samples every capsule off the SAME crystal at exactly
        # OUTPUT_RATE, so there is nothing to convert and nothing to correct.
        # Running libsamplerate at ratio 1.0 would still band-limit the signal
        # and, far worse, let the drift loop trim each channel's ratio
        # independently: that injects a per-channel sub-sample delay and
        # destroys precisely the inter-channel phase the beamformer and
        # doa_mod live on. 32 mm at 343 m/s is 93 us end to end, so a 0.1 %
        # ratio error is a whole bearing's worth of delay within a second.
        if int(round(input_rate)) == int(round(output_rate)):
            self.passthrough = True
            self.backend = "passthrough (1:1, single clock)"
            return

        try:
            import samplerate as _libsr
            # sinc_fastest: best latency/quality trade for a Pi-class CPU.
            self._r = _libsr.Resampler("sinc_fastest", channels=1)
            self.backend = "libsamplerate(sinc_fastest)"
            self.variable = True
        except Exception:
            from scipy.signal import resample_poly  # noqa: F401
            gcd = math.gcd(int(round(input_rate)), int(round(output_rate)))
            self.up = int(round(output_rate)) // gcd
            self.down = int(round(input_rate)) // gcd
            self.backend = f"scipy.resample_poly({self.up}:{self.down})"

    def process(self, x, ratio_scale=1.0):
        """Resample float32 mono `x`; `ratio_scale` trims the ratio (variable
        backend only) for closed-loop drift correction."""
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.passthrough:
            return x.astype(np.float32, copy=False)
        if self._r is not None:
            ratio = self.base_ratio * ratio_scale
            return self._r.process(
                x.astype(np.float32, copy=False), ratio, end_of_input=False
            )
        from scipy.signal import resample_poly
        return resample_poly(
            x.astype(np.float32, copy=False), self.up, self.down
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# One microphone: capture ring buffer + DC block + async resample to 16 kHz.
# ---------------------------------------------------------------------------
class MicChannel:
    """Owns capture, drift-corrected resampling and the 16 kHz output FIFO for
    a single microphone. All cross-thread state is guarded by `self.lock`."""

    def __init__(self, label, device, sample_rate, name="", hpf_hz=150.0,
                 capture_channels=1):
        self.label = label
        self.device = device
        self.sample_rate = int(sample_rate)
        self.name = name
        self.capture_channels = int(capture_channels)

        self.lock = threading.Lock()
        self.in_buf = np.zeros(0, dtype=np.float32)     # raw capture samples
        self.out_fifo = np.zeros(0, dtype=np.float32)   # resampled 16 kHz
        self.last_capture_mono = None

        self.resampler = Resampler(self.sample_rate, OUTPUT_RATE)
        self.ratio_scale = 1.0
        self.max_in = int(self.sample_rate * MAX_IN_SECONDS)
        self.max_out = MAX_OUT_BLOCKS * L_OUT

        # Speech high-pass applied to the 16 kHz stream. Cheap USB capsules can
        # put out huge sub-200 Hz rumble that swamps the speech band and wrecks
        # the beamformer's per-channel statistics; a 2nd-order Butterworth
        # high-pass (default 150 Hz) removes it while preserving intelligibility
        # (ASR only needs the ~300-3400 Hz band). Designed at OUTPUT_RATE for
        # numerical stability (low normalized cutoff is fine at 16 kHz).
        self.hpf_hz = float(hpf_hz)
        if self.hpf_hz and self.hpf_hz > 0:
            self.hpf_b, self.hpf_a = butter(
                2, self.hpf_hz / (OUTPUT_RATE * 0.5), btype="high")
            self.hpf_zi = lfilter_zi(self.hpf_b, self.hpf_a) * 0.0
        else:
            self.hpf_b = self.hpf_a = self.hpf_zi = None

        # Diagnostics
        self.underruns = 0
        self.in_overflows = 0
        self.out_overflows = 0
        self.xruns = 0

    def push(self, samples, capture_mono):
        """Called from the PortAudio callback. Append raw capture, bounding the
        buffer so a stalled worker cannot grow memory without limit. Filtering
        is done once at 16 kHz in resample_in (cheaper and numerically nicer
        than at the 48 kHz input rate)."""
        with self.lock:
            self.in_buf = np.concatenate(
                (self.in_buf, samples.astype(np.float32, copy=False)))
            if self.in_buf.size > self.max_in:
                self.in_buf = self.in_buf[-self.max_in:]
                self.in_overflows += 1
            self.last_capture_mono = capture_mono

    def resample_in(self):
        """Resample everything captured since the last tick into the 16 kHz
        FIFO, high-pass to strip sub-band rumble, bounding it so latency can
        never run away."""
        with self.lock:
            x = self.in_buf
            self.in_buf = np.zeros(0, dtype=np.float32)

        y = self.resampler.process(x, self.ratio_scale)
        if y.size and self.hpf_b is not None:
            yf, self.hpf_zi = lfilter(
                self.hpf_b, self.hpf_a, y.astype(np.float64), zi=self.hpf_zi)
            y = yf.astype(np.float32)
        if y.size:
            self.out_fifo = np.concatenate((self.out_fifo, y))

        if len(self.out_fifo) > self.max_out:
            self.out_fifo = self.out_fifo[-self.max_out:]
            self.out_overflows += 1

    def update_ratio(self, ref_len):
        """Trim the resample ratio to pull this channel's FIFO toward the
        cross-channel reference, locking the independent USB clocks to each
        other. Absolute FIFO depth is regulated by the consumer, not here, so
        a uniform offset shared by all channels leaves the ratio at 1.0."""
        if not self.resampler.variable:
            return
        err = (len(self.out_fifo) - ref_len) / float(TARGET_OUT)
        corr = max(-RATIO_TRIM_MAX, min(RATIO_TRIM_MAX, -RATIO_TRIM_KP * err))
        self.ratio_scale = 1.0 + corr

    def fifo_len(self):
        return len(self.out_fifo)

    def trim_fifo(self, keep):
        """Drop the oldest samples so the FIFO holds exactly `keep` (used once
        at prime time to phase-align all channels to a common latency)."""
        if len(self.out_fifo) > keep:
            self.out_fifo = self.out_fifo[-keep:]

    def pop_block(self):
        """Return exactly L_OUT samples; pad with zeros on underrun so a slow
        or dead mic degrades gracefully instead of stalling the array."""
        if len(self.out_fifo) >= L_OUT:
            out = self.out_fifo[:L_OUT]
            self.out_fifo = self.out_fifo[L_OUT:]
            return out
        out = np.concatenate(
            (self.out_fifo, np.zeros(L_OUT - len(self.out_fifo), dtype=np.float32))
        )
        self.out_fifo = np.zeros(0, dtype=np.float32)
        self.underruns += 1
        return out


# ---------------------------------------------------------------------------
# Device discovery — prefer external USB inputs, fall back to internal.
# ---------------------------------------------------------------------------
def is_usable_input(name: str, include_internal: bool) -> bool:
    exclude = (
        "blackhole", "loopback", "zoom", "teams", "microsoft",
        "obs", "background music", "monitor", "pulse", "default", "sysdefault",
    )
    low = name.lower()
    if any(t in low for t in exclude):
        return False
    if not include_internal and any(
        t in low for t in ("macbook", "built-in", "internal", "hdmi")
    ):
        return False
    return True


def looks_usb(name: str) -> bool:
    low = name.lower()
    return any(t in low for t in ("usb", "mic", "uac", "card"))


def choose_rate(device: int, requested_rate=None, channels=1) -> int:
    info = sd.query_devices(device)
    default_rate = int(info["default_samplerate"])
    candidates = []
    if requested_rate and requested_rate > 0:
        candidates.append(int(requested_rate))
    candidates += [48000, 44100, 32000, 16000, default_rate]
    seen = set()
    for rate in candidates:
        if rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(
                device=device, channels=channels, dtype="float32", samplerate=rate
            )
            return int(rate)
        except Exception:
            continue
    return default_rate


def detect_microphones(logger, include_internal=False, max_mics=3,
                       requested_rate=None, prefer_usb=True,
                       capture_channels=0, mic_name_filter=""):
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0]

    usable = [(i, d) for i, d in inputs
              if is_usable_input(d["name"], include_internal)]
    logger.info(f"Found {len(usable)} usable input(s) of {len(inputs)} total.")

    if not usable and not include_internal:
        logger.warning("No external mics; retrying with internal inputs.")
        usable = [(i, d) for i, d in inputs
                  if is_usable_input(d["name"], include_internal=True)]

    if mic_name_filter:
        needle = mic_name_filter.lower()
        filtered = [(i, d) for i, d in usable if needle in d["name"].lower()]
        logger.info(
            f"mic_name_filter='{mic_name_filter}' kept {len(filtered)} of "
            f"{len(usable)} usable input(s)."
        )
        if filtered:
            usable = filtered
        else:
            logger.warning(
                f"mic_name_filter='{mic_name_filter}' matched no devices; "
                f"ignoring the filter."
            )

    if prefer_usb:
        # Stable order: USB-looking devices first, then by device index.
        usable.sort(key=lambda it: (not looks_usb(it[1]["name"]), it[0]))

    resolved = []
    for idx, dev in usable:
        if len(resolved) >= max_mics:
            break
        # Decide how many channels to open. 0 = auto: native count capped at 2
        # (stereo-native mics like the PULUZ PU425 need their native format to
        # deliver full level). >0 forces that exact count. Always clamped to
        # [1, 2]; the callback downmixes whatever we open back to one mono
        # stream, so downstream stays single-channel-per-mic either way.
        native = int(dev.get("max_input_channels", 1))
        cap_ch = native if capture_channels == 0 else capture_channels
        cap_ch = max(1, min(cap_ch, 2))
        rate = choose_rate(idx, requested_rate, channels=cap_ch)
        label = f"mic{len(resolved)}"
        logger.info(
            f"Auto-detected {label}: dev #{idx} '{dev['name']}' @ {rate} Hz "
            f"({cap_ch}ch capture -> mono)"
        )
        resolved.append({
            "label": label, "device": idx,
            "sample_rate": rate, "name": dev["name"],
            "capture_channels": cap_ch,
        })

    if not resolved:
        raise RuntimeError("No usable microphone devices could be resolved.")
    return resolved


# ---------------------------------------------------------------------------
# Array mode — ONE multichannel device whose channels are the array mics.
# ---------------------------------------------------------------------------
def parse_channel_list(spec):
    """Parse "1,2,3,4" into [1, 2, 3, 4]. Empty/blank -> []."""
    spec = str(spec or "").strip()
    if not spec:
        return []
    out = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def resolve_array_device(logger, name_filter, channels, device_channels=0,
                         requested_rate=None):
    """Resolve ONE multichannel capture device into M array mics.

    This is the ReSpeaker path, and it exists because the auto-detect path
    above cannot express it. detect_microphones() treats every *device* as one
    microphone and downmixes whatever it opens to mono (capped at 2 channels),
    which is right for N separate USB lavaliers and exactly wrong for a
    ReSpeaker 4 Mic Array: that is ONE UAC1.0 device carrying six channels at
    a fixed 16 kHz, and downmixing it collapses all four capsules into a
    single stream, leaving the beamformer running at M=1 with nothing to
    beam-form and doa_mod with no baseline to measure a delay across.

    So: open the device ONCE at its native width and hand each requested
    channel index to its own MicChannel. One device, one converter, one
    clock — the drift loop is switched off in this mode (see
    VoiceMod._single_clock) because there is no second crystal to lock to.

    ReSpeaker channel map is `FL FR FC LFE SL SR`:
        ch0  processed mix (the board's own XMOS beamform/AEC/NS)
        ch1..ch4  the raw capsules  <- these are the array
        ch5  playback reference
    hence the `1,2,3,4` default in docker-compose.yml.

    Returns the same dict shape detect_microphones() returns, plus
    `stream_channel` (index within the opened device) and `n_dev_ch` (how wide
    to open the stream). Every entry shares one `device`.
    """
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0]
    if not inputs:
        raise RuntimeError("array_channels set but no input devices exist.")

    # RESOLVE BY NAME, NEVER BY INDEX. ALSA card numbers on atb are not stable
    # across reboots -- the ReSpeaker was card 1, a reboot handed that slot to
    # the NVidia HDMI codec and the array came back as card 2 -- and the
    # PortAudio index moves with them. Nothing here, and nothing in
    # docker-compose.yml, may pin a card number, an index, or an hw:X,Y.
    #
    # The name to match is the PORTAUDIO name, which is not the ALSA card id:
    # the card id is `ArrayUAC10` but PortAudio calls the same device
    # "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:N,0)". Matching on
    # "ArrayUAC10" therefore finds nothing. Use "ReSpeaker".
    needle = str(name_filter or "").strip().lower()
    if needle:
        cand = [(i, d) for i, d in inputs if needle in d["name"].lower()]
        if not cand:
            # Loud, and with the evidence attached: a wrong-device fallback
            # that quietly captured the onboard analog jack would look exactly
            # like a broken beamformer on demo day.
            raise RuntimeError(
                f"array_device='{name_filter}' matched no input device, so "
                f"voice_mod is refusing to start rather than capturing some "
                f"other card by accident. Available inputs were: "
                + "; ".join(f"#{i} '{d['name']}' ({d['max_input_channels']}ch)"
                            for i, d in inputs)
                + ". Note that the filter matches the PORTAUDIO device name, "
                  "not the ALSA card id: for the ReSpeaker use 'ReSpeaker', "
                  "not 'ArrayUAC10'.")
    else:
        # No name given: take the widest REAL input device. The same exclusion
        # list the auto-detect path uses is load-bearing here, because ALSA's
        # `default` and `sysdefault` advertise 128 input channels and would
        # otherwise win every time -- and they are software mixers, not the
        # array, so the "capsules" behind them would be silence.
        cand = [(i, d) for i, d in inputs
                if is_usable_input(d["name"], include_internal=True)]
        cand.sort(key=lambda it: -int(it[1]["max_input_channels"]))

    # Prefer a candidate wide enough for the requested channels; among those,
    # the first (name filter order / widest).
    need = max(channels) + 1
    wide = [(i, d) for i, d in cand
            if int(d.get("max_input_channels", 0)) >= need]
    if not wide:
        raise RuntimeError(
            f"array_channels={channels} needs at least {need} device channels; "
            f"candidates offer "
            f"{[(d['name'], d['max_input_channels']) for _, d in cand]}")
    idx, dev = wide[0]
    if not needle:
        # Auto-picked. Say so at WARNING with the whole shortlist, because the
        # silent version of this line is the one that captures the dock jack
        # on demo day and looks like a dead beamformer.
        logger.warning(
            f"array_device was not set, so voice_mod auto-picked the widest "
            f"input device: #{idx} '{dev['name']}' ({dev['max_input_channels']}ch). "
            f"Other candidates were: "
            + "; ".join(f"#{i} '{d['name']}' ({d['max_input_channels']}ch)"
                        for i, d in wide[1:]) + ". Pin it with "
            f"-p array_device:=<name substring> if that is not the array.")

    native = int(dev.get("max_input_channels", 1))
    n_dev_ch = native if device_channels <= 0 else int(device_channels)
    n_dev_ch = max(need, min(n_dev_ch, native))
    rate = choose_rate(idx, requested_rate, channels=n_dev_ch)
    logger.info(
        f"Array device #{idx} '{dev['name']}': opening {n_dev_ch}ch @ {rate} Hz, "
        f"using channels {channels} as mic0..mic{len(channels) - 1}")

    return [
        {"label": f"mic{k}", "device": idx, "sample_rate": rate,
         "name": f"{dev['name']} ch{c}", "capture_channels": 1,
         "stream_channel": int(c), "n_dev_ch": n_dev_ch}
        for k, c in enumerate(channels)
    ]


# ---------------------------------------------------------------------------
# dorai_beamformer.ort beamformer wrapper.
# ---------------------------------------------------------------------------
class Beamformer:
    """Multichannel enhancer. dorai_beamformer.ort is a *whole-signal* model: it takes the
    entire [M, T] utterance in one pass and returns the clean [T] mono — it is
    NOT a streaming model. Feeding it short isolated windows (e.g. 3200/1600)
    and stitching the centers produces a discontinuity at every window boundary
    (an audible click train at the block rate) and starves the model of the
    context its internal STFT/Wiener stage needs. So we enhance one full publish
    frame per call, exactly like the reference app."""

    def __init__(self, model_path, num_threads, logger):
        so = ort.SessionOptions()
        so.log_severity_level = 3          # errors only (suppress GPU-probe warns)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads and num_threads > 0:
            so.intra_op_num_threads = int(num_threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.logger = logger

    def warmup(self, num_channels):
        dummy = np.zeros((num_channels, WINDOW_SIZE), dtype=np.float32)
        for _ in range(2):
            self.sess.run([self.out_name], {self.in_name: dummy})

    def enhance_frame(self, mc):
        """Enhance one [M, T] multichannel frame in a single pass.

        Peak-normalize the whole frame (the model expects roughly unit-scale
        input, as the reference app does), then restore the original scale on
        the output so per-frame loudness is preserved and silent frames don't
        get their noise amplified."""
        x = np.ascontiguousarray(mc.astype(np.float32, copy=False))
        L = x.shape[1]
        g = float(np.max(np.abs(x)))
        xn = x / g if g > 1e-9 else x
        try:
            out = self.sess.run([self.out_name], {self.in_name: xn})[0]
            clean = np.asarray(out, dtype=np.float32).reshape(-1)[:L]
            if g > 1e-9:
                clean = clean * g
        except Exception as e:  # never let inference kill the stream
            self.logger.error(
                f"Inference failed: {e}", throttle_duration_sec=2.0)
            clean = np.zeros(L, dtype=np.float32)
        return clean


# ---------------------------------------------------------------------------
# ROS 2 node.
# ---------------------------------------------------------------------------
class VoiceMod(Node):
    def __init__(self):
        super().__init__("voice_mod")

        # Float params use dynamic typing so both `p:=10` (int) and `p:=10.0`
        # (double) are accepted on the command line without a type error.
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter("include_internal", False)
        self.declare_parameter("max_mics", 3)
        self.declare_parameter("requested_rate", 48000)
        self.declare_parameter("prefer_usb", True)
        # Case-insensitive substring; when set, only input devices whose name
        # contains it are used as array mics. Lets you pin the real array (e.g.
        # "earpod") and exclude dead/irrelevant inputs like an empty dock jack
        # that would otherwise be auto-selected and collapse the beamformer.
        self.declare_parameter("mic_name_filter", "")
        # Per-mic capture channel count. 0 = auto (open each device at its native
        # channel count, capped at 2, then downmix to mono); 1 = force the legacy
        # mono-open path; 2 = force stereo. Some USB-C lavaliers (e.g. PULUZ
        # PU425) are stereo-native and deliver a ~3x weaker signal when opened
        # mono, so auto restores full level while staying mono downstream.
        self.declare_parameter("capture_channels", 0)
        # --- Array mode (ReSpeaker and any other single multichannel board) --
        # Set array_channels to a comma-separated list of DEVICE channel
        # indices and voice_mod stops hunting for N separate USB mics: it opens
        # ONE device and treats those channels as the array. For the ReSpeaker
        # 4 Mic Array (UAC1.0, 6ch, 16 kHz fixed, map FL FR FC LFE SL SR) the
        # answer is "1,2,3,4" -- ch0 is the board's own processed mix and ch5
        # is the playback reference, neither of which is a capsule.
        # Kept as a STRING rather than an int array so `-p array_channels:=""`
        # cleanly means "off" (an empty ROS 2 integer-array parameter cannot be
        # declared without an explicit descriptor type) and so the compose file
        # stays readable.
        self.declare_parameter("array_channels", "")
        # Case-insensitive device-name substring picking the array device.
        # Empty = take the widest input device, which is the array on any host
        # that has one. Only consulted when array_channels is set.
        self.declare_parameter("array_device", "")
        # How many channels to open on that device. 0 = the device's native
        # width (6 on the ReSpeaker). Never opens fewer than max(channels)+1.
        self.declare_parameter("array_device_channels", 0)
        self.declare_parameter("publish_interval", 10.0, dyn)   # N-second frame
        self.declare_parameter("output_topic", "/dorai_clean_audio")
        # On a 4-core Pi 4B, letting ORT grab all cores starves the PortAudio
        # capture threads and causes input overflows. 2 leaves headroom.
        self.declare_parameter("num_threads", 2)           # 0 = ORT default
        # PortAudio input latency (s). Larger = bigger OS capture buffer = more
        # tolerance to worker/CPU jitter before an overflow. <=0 => 'high'.
        self.declare_parameter("input_latency", 0.0, dyn)
        # Debug: also publish the pre-beamformer multichannel audio so the
        # recorder can capture raw + clean simultaneously, time-aligned.
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("raw_topic", "/dorai_raw_audio")
        # Speech high-pass cutoff (Hz) per mic. Raise toward ~200 if low-freq
        # rumble from cheap USB mics still dominates; 0 disables.
        self.declare_parameter("hpf_hz", 150.0, dyn)
        # Output loudness: normalize each published clean frame up to this peak
        # so the speech is easy to hear / gives STT a healthy level. 0 disables
        # (keep dorai_beamformer's native scale). output_max_gain caps the boost so silent
        # frames aren't amplified into loud noise.
        self.declare_parameter("output_peak", 0.9, dyn)
        self.declare_parameter("output_max_gain", 12.0, dyn)
        # Debug: per-frame capture -> clean-speech delay metering. Setting a
        # non-empty path auto-enables metering (no separate boolean needed)
        # and appends one CSV row per published frame:
        #   seq,capture_epoch,clean_ready_epoch,delay_ms,samples,frame_s
        # start_time = wall-clock when the frame's first raw sample was
        # captured (mic ADC time); end_time = wall-clock right after dorai_beamformer
        # enhancement finishes for that frame (pre-publish). The delta is
        # dominated by the whole-signal buffering wait (~publish_interval)
        # plus inference time — useful for confirming the pipeline stays
        # within its expected latency budget on the Pi.
        self.declare_parameter("delay_log_path", "")
        self.declare_parameter("model_path", self._default_model_path())

        gp = self.get_parameter
        self.include_internal = gp("include_internal").value
        self.max_mics = int(gp("max_mics").value)
        req_rate = int(gp("requested_rate").value)
        self.requested_rate = req_rate if req_rate > 0 else None
        self.prefer_usb = bool(gp("prefer_usb").value)
        self.mic_name_filter = str(gp("mic_name_filter").value or "").strip()
        self.capture_channels = int(gp("capture_channels").value)
        self.array_channels = parse_channel_list(gp("array_channels").value)
        self.array_device = str(gp("array_device").value or "").strip()
        self.array_device_channels = int(gp("array_device_channels").value)
        self.publish_interval = float(gp("publish_interval").value)
        self.output_topic = gp("output_topic").value
        num_threads = int(gp("num_threads").value)
        in_lat = float(gp("input_latency").value)
        self.input_latency = in_lat if in_lat > 0 else "high"
        self.publish_raw = bool(gp("publish_raw").value)
        self.raw_topic = gp("raw_topic").value
        self.hpf_hz = float(gp("hpf_hz").value)
        self.output_peak = float(gp("output_peak").value)
        self.output_max_gain = float(gp("output_max_gain").value)
        self.delay_log_path = str(gp("delay_log_path").value or "").strip()
        model_path = gp("model_path").value

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"dorai_beamformer model not found: {model_path}")

        self.get_logger().info(
            f"voice_mod starting: model={model_path}, "
            f"publish_interval={self.publish_interval:.1f}s"
        )

        # Array mode (one multichannel board) or the legacy N-separate-devices
        # auto-detect. `_single_clock` is what the rest of the node keys off:
        # it means every channel came off one converter, so the per-mic
        # drift-correcting resample loop has nothing to correct and is skipped.
        if self.array_channels:
            resolved = resolve_array_device(
                self.get_logger(), self.array_device or self.mic_name_filter,
                self.array_channels, self.array_device_channels,
                self.requested_rate,
            )
            self._single_clock = True
            # One lock for the WHOLE array. The four capsules come off one
            # stream, so they must always hold exactly the same number of
            # samples; MicChannel's per-channel locks cannot guarantee that,
            # because the worker's resample_in() loop can interleave with the
            # callback's push() loop and drain some channels for a block the
            # others have not been given yet. That leaves one capsule a whole
            # 100 ms block ahead FOREVER -- observed on a real start, mic0 at
            # fifo=4800 against 3200 on the rest -- and 100 ms is ~1600
            # samples against a gcc_phat max_tau of 5, so every pair touching
            # that capsule decoheres and the bearing is built from a channel a
            # tenth of a second out. The lock is held for a passthrough plus a
            # 2nd-order filter over 1600 samples x 4, i.e. tens of
            # microseconds against a 100 ms callback period.
            self._array_lock = threading.Lock()
        else:
            resolved = detect_microphones(
                self.get_logger(), self.include_internal, self.max_mics,
                self.requested_rate, self.prefer_usb, self.capture_channels,
                self.mic_name_filter,
            )
            self._single_clock = False
            self._array_lock = None
        self.mics = [
            MicChannel(m["label"], m["device"], m["sample_rate"], m["name"],
                       hpf_hz=self.hpf_hz, capture_channels=m["capture_channels"])
            for m in resolved
        ]
        for ch, m in zip(self.mics, resolved):
            ch.stream_channel = m.get("stream_channel")
        self._array_device = resolved[0]["device"] if self._single_clock else None
        self._array_n_dev_ch = resolved[0].get("n_dev_ch") if self._single_clock else None
        self.num_channels = len(self.mics)

        self.beam = Beamformer(model_path, num_threads, self.get_logger())
        self.beam.warmup(self.num_channels)
        self.get_logger().info(
            f"dorai_beamformer.ort ready ({self.num_channels}ch) via {self.beam.in_name}"
            f"->{self.beam.out_name}"
        )

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.diag_pub = self.create_publisher(String, "/voice_mod/diagnostics", 10)

        # Lightweight per-mic level tap for the demo dashboard: publishes each
        # microphone's post-resample, pre-beamform RMS on a fast cadence
        # (default every 0.2 s) so a UI can render live VU meters without
        # shipping any audio. Cheap — one RMS per popped block.
        self.declare_parameter("levels_topic", "/voice_mod/levels")
        self.declare_parameter("levels_period", 0.2, dyn)
        self.levels_topic = self.get_parameter("levels_topic").value
        self.levels_period = float(self.get_parameter("levels_period").value)
        self.levels_pub = self.create_publisher(
            Float32MultiArray, self.levels_topic, 10)
        self._last_levels = time.monotonic()
        self._levels_val = [0.0] * self.num_channels

        # Fast raw tap for the demo dashboard's live "before" spectrogram: a
        # down-mixed mono chunk published every raw_fast_period (~8 Hz),
        # decoupled from the whole-signal beamformer frame so the raw panel can
        # scroll smoothly. Viz-only and mono; per-mic playback still uses the
        # full-rate multichannel /dorai_raw_audio frame. Off by default.
        self.declare_parameter("publish_raw_fast", False)
        self.declare_parameter("raw_fast_topic", "/dorai_raw_fast")
        # Default below the 0.1 s worker tick so a chunk goes out every tick
        # (~10 Hz). Raise it on a CPU-tight Pi to trade smoothness for load.
        self.declare_parameter("raw_fast_period", 0.05, dyn)
        self.publish_raw_fast = bool(self.get_parameter("publish_raw_fast").value)
        self.raw_fast_topic = self.get_parameter("raw_fast_topic").value
        self.raw_fast_period = float(self.get_parameter("raw_fast_period").value)
        self.raw_fast_pub = (
            self.create_publisher(Float32MultiArray, self.raw_fast_topic, 10)
            if self.publish_raw_fast else None)
        self._fast_acc = []              # mono blocks pending fast publish
        self._last_fast = time.monotonic()
        self._fast_seq = 0

        # Fast MULTICHANNEL raw tap. This is doa_mod's input, and it exists
        # because neither existing tap can serve it: /dorai_raw_audio is
        # multichannel but arrives one publish_interval at a time (2 Hz at the
        # demo's 0.5 s frames, too slow for a 10 Hz bearing), and
        # /dorai_raw_fast is fast but mono-downmixed, which destroys the very
        # inter-channel delays a direction estimate is made of. Same 4-float
        # header and interleaving as /dorai_raw_audio, just smaller and more
        # often. The ReSpeaker is a single exclusive ALSA capture endpoint, so
        # this tap is also the ONLY way doa_mod can get array audio at all
        # while voice_mod holds the device.
        self.declare_parameter("publish_raw_mc", False)
        self.declare_parameter("raw_mc_topic", "/dorai_raw_mc")
        # 0.1 s = the worker tick, so exactly one block per publish and a
        # 10 Hz cadence, which is the DEMO_INTERFACES rate cap for /dorai/doa.
        self.declare_parameter("raw_mc_period", 0.1, dyn)
        self.publish_raw_mc = bool(self.get_parameter("publish_raw_mc").value)
        self.raw_mc_topic = self.get_parameter("raw_mc_topic").value
        self.raw_mc_period = float(self.get_parameter("raw_mc_period").value)
        self.raw_mc_pub = (
            self.create_publisher(Float32MultiArray, self.raw_mc_topic, 10)
            if self.publish_raw_mc else None)
        self._mc_fast_acc = []           # [M, L_OUT] blocks pending
        self._mc_fast_cap0 = None        # capture_s of the oldest pending block
        self._last_mc_fast = time.monotonic()
        self._mc_fast_seq = 0

        self.raw_pub = None
        if self.publish_raw:
            self.raw_pub = self.create_publisher(
                Float32MultiArray, self.raw_topic, 10)

        self._t0 = time.monotonic()
        self._t0_wall = time.time()
        self._delay_log_file = self._open_delay_log(self.delay_log_path)
        self._seq = 0
        self._raw_seq = 0
        self._acc_lock = threading.Lock()
        self._mc_acc = []                    # raw [M, L_OUT] blocks pending
        self._mc_caps = []                   # capture time per pending block
        self._frame_blocks = max(
            1, round(self.publish_interval * OUTPUT_RATE / L_OUT))
        self._last_diag = time.monotonic()
        self._primed = False
        self._running = True

        # Building a ~160k-element Python list and serializing it over DDS is
        # slow; doing it inline would stall the resample/inference loop long
        # enough to overflow the mic buffers. Hand finished frames to a separate
        # publisher thread so the worker keeps draining the capture buffers.
        self._pubq = queue.Queue(maxsize=8)
        self._pub_thread = threading.Thread(
            target=self._pub_loop, name="voice-pub", daemon=True)
        self._pub_thread.start()

        if self._single_clock:
            # ONE stream for the whole array. Opening the device once per
            # capsule would fail anyway (a UAC1.0 array is a single exclusive
            # ALSA capture endpoint), and it is the shared open that gives
            # doa_mod its sample-accurate inter-channel alignment.
            rate = self.mics[0].sample_rate
            blocksize = int(rate * BLOCK_S)
            self._shared_stream = sd.InputStream(
                samplerate=rate, device=self._array_device,
                channels=self._array_n_dev_ch,
                blocksize=blocksize, dtype="float32",
                latency=self.input_latency,
                callback=self._make_array_callback(),
            )
            self._shared_stream.start()
            self.get_logger().info(
                f"Opened array dev#{self._array_device} @ {rate} Hz "
                f"({self._array_n_dev_ch}ch, blk {blocksize}) -> "
                f"{[ch.stream_channel for ch in self.mics]} via "
                f"{self.mics[0].resampler.backend}"
            )
        else:
            self._shared_stream = None
            for ch in self.mics:
                blocksize = int(ch.sample_rate * BLOCK_S)
                ch.stream = sd.InputStream(
                    samplerate=ch.sample_rate, device=ch.device,
                    channels=ch.capture_channels,
                    blocksize=blocksize, dtype="float32",
                    latency=self.input_latency,
                    callback=self._make_callback(ch),
                )
                ch.stream.start()
                self.get_logger().info(
                    f"Opened {ch.label} dev#{ch.device} @ {ch.sample_rate} Hz "
                    f"(blk {blocksize}) via {ch.resampler.backend}"
                )

        self._worker = threading.Thread(
            target=self._run, name="voice-worker", daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"Publishing {self.output_topic}: 1ch @ {OUTPUT_RATE} Hz, "
            f"batched every {self.publish_interval:.1f}s"
        )

    def _open_delay_log(self, path):
        """Open the delay-metering log if `path` is set (auto-enable; no
        separate boolean flag). Returns the open file handle, or None if
        metering is disabled or the path can't be opened."""
        if not path:
            return None
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", buffering=1)  # line-buffered
            if new_file:
                f.write(
                    "seq,capture_epoch,frame_ready_epoch,dequeue_epoch,"
                    "clean_ready_epoch,publish_done_epoch,"
                    "buffer_ms,queue_ms,infer_ms,publish_ms,total_ms,"
                    "samples,frame_s\n"
                )
            self.get_logger().info(f"Delay metering ENABLED -> {path}")
            return f
        except Exception as e:
            self.get_logger().error(
                f"Could not open delay_log_path '{path}': {e}. "
                f"Delay metering disabled."
            )
            return None

    def _log_delay(self, cap0, seq, n_samples, frame_ready_mono, dequeue_mono,
                    infer_done_mono, publish_done_mono):
        """Append one CSV row breaking the per-frame delay into stages:

          buffer_ms  - capture -> frame fully accumulated (mandatory wait,
                       ~= publish_interval; whole-signal model needs the
                       full frame before it can run)
          queue_ms   - frame accumulated -> publisher thread picked it up
                       (should be ~0; growth means the pub thread is still
                       busy on the previous frame's inference, i.e. falling
                       behind real time)
          infer_ms   - dequeue -> dorai_beamformer enhancement done (includes the
                       optional publish_raw tap + array concat, but
                       dominated by ONNX inference)
          publish_ms - enhancement done -> ROS publish call returns
                       (float->list serialization + DDS send)
          total_ms   - capture -> publish done (buffer+queue+infer+publish)
        """
        if self._delay_log_file is None:
            return
        capture_mono = self._t0 + cap0
        buffer_ms = (frame_ready_mono - capture_mono) * 1000.0
        queue_ms = (dequeue_mono - frame_ready_mono) * 1000.0
        infer_ms = (infer_done_mono - dequeue_mono) * 1000.0
        publish_ms = (publish_done_mono - infer_done_mono) * 1000.0
        total_ms = (publish_done_mono - capture_mono) * 1000.0

        capture_epoch = self._t0_wall + cap0
        frame_ready_epoch = self._t0_wall + (frame_ready_mono - self._t0)
        dequeue_epoch = self._t0_wall + (dequeue_mono - self._t0)
        clean_ready_epoch = self._t0_wall + (infer_done_mono - self._t0)
        publish_done_epoch = self._t0_wall + (publish_done_mono - self._t0)
        frame_s = n_samples / float(OUTPUT_RATE)

        try:
            self._delay_log_file.write(
                f"{seq},{capture_epoch:.3f},{frame_ready_epoch:.3f},"
                f"{dequeue_epoch:.3f},{clean_ready_epoch:.3f},"
                f"{publish_done_epoch:.3f},{buffer_ms:.1f},{queue_ms:.1f},"
                f"{infer_ms:.1f},{publish_ms:.1f},{total_ms:.1f},"
                f"{n_samples},{frame_s:.3f}\n"
            )
        except Exception as e:
            self.get_logger().error(
                f"delay log write failed: {e}", throttle_duration_sec=5.0)
        self.get_logger().info(
            f"delay seq={seq}: total={total_ms:.0f}ms "
            f"(buffer={buffer_ms:.0f} queue={queue_ms:.0f} "
            f"infer={infer_ms:.0f} publish={publish_ms:.0f})",
            throttle_duration_sec=1.0,
        )

    def _default_model_path(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            p = os.path.join(get_package_share_directory("voice_mod"), "dorai_beamformer.ort")
            if os.path.exists(p):
                return p
        except Exception:
            pass
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dorai_beamformer.ort")
        return local if os.path.exists(local) else "voice_mod/dorai_beamformer.ort"

    def _make_callback(self, ch):
        def callback(indata, frames, time_info, status):
            if status:
                ch.xruns += 1
                self.get_logger().warning(
                    f"{ch.label} xrun ({status}); total={ch.xruns}. If frequent, "
                    f"raise input_latency or lower num_threads.",
                    throttle_duration_sec=5.0
                )
            now = time.monotonic()
            try:
                age = time_info.currentTime - time_info.inputBufferAdcTime
                capture_mono = now - max(age, 0.0)
            except Exception:
                capture_mono = now
            # Collapse whatever we captured to one mono stream. For a genuine
            # mono device this is exactly the old indata[:, 0]; for a stereo
            # capture it averages the channels (identical channels -> full
            # level; signal on one channel -> still preserved).
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0]
            ch.push(mono.copy(), capture_mono)
        return callback

    def _make_array_callback(self):
        """One PortAudio callback fanning a single multichannel capture out to
        the per-capsule MicChannels. No downmix, no per-channel resample: the
        channels are already on one clock at one rate, and slicing preserves
        the sample-exact alignment the TDOA estimate depends on."""
        def callback(indata, frames, time_info, status):
            if status:
                for ch in self.mics:
                    ch.xruns += 1
                self.get_logger().warning(
                    f"array xrun ({status}); total={self.mics[0].xruns}. If "
                    f"frequent, raise input_latency or lower num_threads.",
                    throttle_duration_sec=5.0,
                )
            now = time.monotonic()
            try:
                age = time_info.currentTime - time_info.inputBufferAdcTime
                capture_mono = now - max(age, 0.0)
            except Exception:
                capture_mono = now
            with self._array_lock:
                for ch in self.mics:
                    ch.push(indata[:, ch.stream_channel].copy(), capture_mono)
        return callback

    def _run(self):
        next_tick = time.monotonic()
        while self._running:
            # Pace to a steady BLOCK_S grid without accumulating drift.
            next_tick += BLOCK_S
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()   # we fell behind; resync

            try:
                self._tick()
            except Exception as e:
                self.get_logger().error(
                    f"worker tick error: {e}", throttle_duration_sec=2.0
                )

    def _tick(self):
        if self._array_lock is not None:
            # Atomic against the array callback: see _make_array_callback.
            with self._array_lock:
                for ch in self.mics:
                    ch.resample_in()
        else:
            for ch in self.mics:
                ch.resample_in()
        # Lock the independent USB clocks to each other by pulling every
        # channel's FIFO toward the cross-channel mean. Skipped entirely in
        # array mode: one converter means the FIFOs advance in lockstep by
        # construction, so the loop would only be reacting to its own
        # quantisation noise -- and each trim it applied would resample one
        # capsule relative to the others, which is a bearing error, not a
        # correction. (The resampler is a 1:1 passthrough there too, so
        # update_ratio() is already a no-op; this makes the intent explicit.)
        if len(self.mics) > 1 and not self._single_clock:
            ref = sum(ch.fifo_len() for ch in self.mics) / len(self.mics)
            for ch in self.mics:
                ch.update_ratio(ref)

        # On first fill, trim all FIFOs to a common depth so the channels
        # start phase-aligned before enhancement begins.
        if not self._primed:
            depths = [ch.fifo_len() for ch in self.mics]
            if min(depths) >= max(TARGET_OUT, L_OUT):
                common = min(depths)
                for ch in self.mics:
                    ch.trim_fifo(common)
                self._primed = True
                self.get_logger().info("FIFOs primed and aligned; enhancing.")
            else:
                return

        # Consume every aligned block above the alignment cushion, so the
        # output rate tracks the true capture rate even when the worker tick
        # jitters (an inference spike can let two input blocks pile up). Popping
        # the same count from every channel, gated by the slowest, keeps them
        # phase-aligned; the cushion absorbs brief input starvation.
        # Accumulate phase-aligned multichannel blocks at the true capture rate.
        # The worker stays light (resample + pop + append); the heavy dorai_beamformer pass
        # happens on the publisher thread so it never starves the mic buffers.
        min_fifo = min(ch.fifo_len() for ch in self.mics)
        n_blocks = max(0, (min_fifo - TARGET_OUT) // L_OUT)
        for _ in range(int(n_blocks)):
            block = np.stack([ch.pop_block() for ch in self.mics], axis=0)
            # Per-mic RMS for the live level meters (latest block wins).
            self._levels_val = np.sqrt(
                np.mean(block.astype(np.float64) ** 2, axis=1)).tolist()
            # Down-mixed mono block for the fast raw-spectrogram tap.
            if self.raw_fast_pub is not None:
                self._fast_acc.append(block.mean(axis=0).astype(np.float32))
            with self.mics[0].lock:
                cap = self.mics[0].last_capture_mono
            capture_s = (cap or time.monotonic()) - self._t0
            if self.raw_mc_pub is not None:
                if not self._mc_fast_acc:
                    self._mc_fast_cap0 = capture_s
                self._mc_fast_acc.append(block)
            with self._acc_lock:
                self._mc_acc.append(block)
                self._mc_caps.append(capture_s)

        # Hand each complete publish_interval-second multichannel frame to the
        # publisher thread for whole-frame enhancement + publish.
        while True:
            with self._acc_lock:
                if len(self._mc_acc) < self._frame_blocks:
                    break
                blocks = self._mc_acc[:self._frame_blocks]
                cap0 = self._mc_caps[0]
                del self._mc_acc[:self._frame_blocks]
                del self._mc_caps[:self._frame_blocks]
            self._enqueue((blocks, cap0, time.monotonic()))

        now = time.monotonic()
        if now - self._last_levels >= self.levels_period:
            self._last_levels = now
            lv = Float32MultiArray()
            lv.data = [float(v) for v in self._levels_val]
            self.levels_pub.publish(lv)

        # Fast raw tap: flush the accumulated mono down-mix as one small chunk.
        if (self.raw_fast_pub is not None and self._fast_acc
                and now - self._last_fast >= self.raw_fast_period):
            self._last_fast = now
            chunk = np.concatenate(self._fast_acc)
            self._fast_acc = []
            fmsg = Float32MultiArray()
            header = np.array(
                [0.0, float(OUTPUT_RATE), 1.0, float(self._fast_seq)],
                dtype=np.float32)
            fmsg.data = np.concatenate((header, chunk)).tolist()
            fmsg.layout.data_offset = 4
            self.raw_fast_pub.publish(fmsg)
            self._fast_seq += 1

        # Fast multichannel tap: flush the pending blocks as one interleaved
        # frame. Deliberately silent (no per-frame log line like _emit_raw's)
        # because at 10 Hz that would drown the diagnostics.
        # The window is 10 percent short of the nominal period on purpose. The
        # worker already ticks on a BLOCK_S grid, so one block is ready every
        # period; a hard `>= period` test fails on every tick that runs a hair
        # early, and the block it skipped is then published together with the
        # next one. Measured on the ReSpeaker that turned a 10 Hz tap into a
        # 6.4 Hz one carrying 3200 samples at a time, which in turn cost
        # doa_mod a third of its output rate.
        if (self.raw_mc_pub is not None and self._mc_fast_acc
                and now - self._last_mc_fast >= self.raw_mc_period * 0.9):
            self._last_mc_fast = now
            mc = np.concatenate(self._mc_fast_acc, axis=1)      # [M, T]
            cap0 = self._mc_fast_cap0 if self._mc_fast_cap0 is not None else 0.0
            self._mc_fast_acc = []
            self._mc_fast_cap0 = None
            m = mc.shape[0]
            interleaved = mc.T.reshape(-1).astype(np.float32)
            mmsg = Float32MultiArray()
            header = np.array(
                [cap0, float(OUTPUT_RATE), float(m), float(self._mc_fast_seq)],
                dtype=np.float32)
            mmsg.data = np.concatenate((header, interleaved)).tolist()
            mmsg.layout.data_offset = 4
            mmsg.layout.dim = [
                MultiArrayDimension(label="channels", size=m,
                                    stride=len(interleaved)),
                MultiArrayDimension(label="samples", size=mc.shape[1],
                                    stride=mc.shape[1]),
            ]
            self.raw_mc_pub.publish(mmsg)
            self._mc_fast_seq += 1

        self._maybe_diag(time.monotonic())

    def _enqueue(self, item):
        """Queue a finished frame for the publisher thread; drop oldest on a
        full queue so a slow consumer can never back-pressure the worker."""
        try:
            self._pubq.put_nowait(item)
        except queue.Full:
            try:
                self._pubq.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pubq.put_nowait(item)
            except queue.Full:
                pass

    def _pub_loop(self):
        """Enhance (whole frame) + serialize + publish, off the worker thread."""
        while self._running:
            try:
                blocks, cap0, frame_ready_mono = self._pubq.get(timeout=0.2)
            except queue.Empty:
                continue
            dequeue_mono = time.monotonic()
            try:
                self._process_frame(blocks, cap0, frame_ready_mono, dequeue_mono)
            except Exception as e:
                self.get_logger().error(
                    f"publish error: {e}", throttle_duration_sec=2.0)

    def _process_frame(self, blocks, cap0, frame_ready_mono=None,
                        dequeue_mono=None):
        """Assemble [M, T], optionally publish the raw tap, enhance, publish.

        `frame_ready_mono`/`dequeue_mono` are only used for delay metering
        (stage breakdown); on the shutdown flush path there's no queueing, so
        they default to "now" at call time.
        """
        now = time.monotonic()
        if frame_ready_mono is None:
            frame_ready_mono = now
        if dequeue_mono is None:
            dequeue_mono = now

        mc = np.concatenate(blocks, axis=1)          # [M, T]
        if self.publish_raw:
            self._emit_raw(mc, cap0)
        clean = self.beam.enhance_frame(mc)
        infer_done_mono = time.monotonic()
        seq = self._seq
        self._emit(clean, cap0)
        publish_done_mono = time.monotonic()
        if self._delay_log_file is not None:
            self._log_delay(
                cap0, seq, len(clean),
                frame_ready_mono, dequeue_mono, infer_done_mono, publish_done_mono,
            )

    def _flush(self):
        """Publish any remaining partial frame (called on shutdown)."""
        with self._acc_lock:
            blocks = self._mc_acc
            cap0 = self._mc_caps[0] if self._mc_caps else 0.0
            self._mc_acc = []
            self._mc_caps = []
        if blocks:
            self._process_frame(blocks, cap0)

    def _emit_raw(self, mc, capture_s):
        """Publish one frame of pre-beamformer multichannel audio (debug tap).
        `mc` is a [M, T] array; channels are interleaved in `data[4:]` exactly
        like the clean topic so the same reader works."""
        if self.raw_pub is None:
            return
        m = mc.shape[0]
        interleaved = mc.T.reshape(-1).astype(np.float32)   # T*M, ch-interleaved
        msg = Float32MultiArray()
        header = np.array(
            [capture_s, float(OUTPUT_RATE), float(m), float(self._raw_seq)],
            dtype=np.float32,
        )
        msg.data = np.concatenate((header, interleaved)).tolist()
        msg.layout.data_offset = 4
        per_ch = mc.shape[1]
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=m, stride=len(interleaved)),
            MultiArrayDimension(label="samples", size=per_ch, stride=per_ch),
        ]
        self.raw_pub.publish(msg)
        self.get_logger().info(
            f"Published RAW frame seq={self._raw_seq}: {m}ch x {per_ch} samples "
            f"({per_ch / OUTPUT_RATE:.2f}s)"
        )
        self._raw_seq += 1

    def _emit(self, clean, capture_s):
        clean = np.asarray(clean, dtype=np.float32)
        # Loudness normalization: lift the frame to a target peak (capped so a
        # near-silent frame isn't blown up), then hard-limit to avoid clipping.
        if self.output_peak > 0:
            peak = float(np.max(np.abs(clean)))
            if peak > 1e-6:
                gain = min(self.output_peak / peak, self.output_max_gain)
                clean = clean * gain
            clean = np.clip(clean, -0.99, 0.99).astype(np.float32)
        msg = Float32MultiArray()
        header = np.array(
            [capture_s, float(OUTPUT_RATE), 1.0, float(self._seq)],
            dtype=np.float32,
        )
        msg.data = np.concatenate((header, clean)).tolist()
        msg.layout.data_offset = 4
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=1, stride=len(clean)),
            MultiArrayDimension(label="samples", size=len(clean), stride=len(clean)),
        ]
        self.pub.publish(msg)
        self.get_logger().info(
            f"Published frame seq={self._seq}: {len(clean)} samples "
            f"({len(clean) / OUTPUT_RATE:.2f}s)"
        )
        self._seq += 1

    def _maybe_diag(self, now):
        if now - self._last_diag < DIAG_PERIOD_S:
            return
        self._last_diag = now
        parts = []
        for ch in self.mics:
            with ch.lock:
                inb = len(ch.in_buf)
            parts.append(
                f"{ch.label}:{ch.sample_rate}Hz fifo={ch.fifo_len():5d} "
                f"in={inb:5d} ur={ch.underruns} xr={ch.xruns} "
                f"ovf={ch.in_overflows + ch.out_overflows} "
                f"rs={ch.ratio_scale:.4f}"
            )
        with self._acc_lock:
            acc = len(self._mc_acc) * L_OUT
        text = " | ".join(parts) + f" || acc={acc} samples"
        # In array mode the depths are equal by construction. If they are not,
        # the capsules are skewed by whole blocks and every bearing downstream
        # is built on a lie, so say so at ERROR rather than leaving it to
        # whoever happens to read a diagnostics line.
        if self._single_clock:
            depths = [ch.fifo_len() for ch in self.mics]
            if max(depths) != min(depths):
                self.get_logger().error(
                    f"CAPSULE SKEW: FIFO depths {depths} differ by "
                    f"{max(depths) - min(depths)} samples "
                    f"({(max(depths) - min(depths)) / OUTPUT_RATE * 1000:.0f} ms). "
                    f"The capsules are no longer sample-aligned and /dorai/doa "
                    f"cannot be trusted. Restart voice_mod.",
                    throttle_duration_sec=10.0)
        self.get_logger().info(text)
        self.diag_pub.publish(String(data=text))

    def shutdown(self):
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        for st in ([getattr(self, "_shared_stream", None)]
                   + [getattr(ch, "stream", None) for ch in self.mics]):
            if st is not None:
                try:
                    st.stop()
                    st.close()
                except Exception:
                    pass
        # flush any tail audio so the last words aren't lost
        try:
            self._flush()
        except Exception:
            pass
        if getattr(self, "_pub_thread", None) is not None \
                and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)
        if getattr(self, "_delay_log_file", None) is not None:
            try:
                self._delay_log_file.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = VoiceMod()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
