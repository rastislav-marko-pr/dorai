#!/usr/bin/env python3
"""
web_mod / web.py  —  dorai demo dashboard (headless-robot friendly).

A single ROS 2 node that bridges the dorai pipeline topics to a browser so the
mic-array capture and beamformer can be demonstrated on a robot with no screen.
Point a laptop/phone on the same network at http://<robot-ip>:8080.

What it shows / does:
  * live per-mic level meters               (/voice_mod/levels)
  * raw multichannel vs. clean spectrogram   (/dorai_raw_audio, /dorai_clean_audio)
    + waveform envelopes  -> the beamformer noise-suppression, made visible
  * streaming transcript (partials + finals) (/dorai_partial_transcript, /dorai_transcript)
  * per-mic drift / xrun health              (/voice_mod/diagnostics)
  * "Play raw" / "Play clean" buttons that audition the last frame through the
    ROBOT's own speakers (server-side sounddevice output) — the client only
    needs a browser, no audio round-trip.

Architecture:
  * rclpy spins on a background thread; topic callbacks compute compact,
    display-ready summaries (decimated waveforms, small spectrograms, RMS) and
    hand them to the asyncio loop via loop.call_soon_threadsafe.
  * aiohttp serves one self-contained index.html and a /ws WebSocket that
    fans out those summaries to every connected browser and receives play/stop
    commands.
  * The most recent raw [M, T] and clean [T] float frames are cached under a
    lock so a Play command can push them straight to the speaker.
"""

import os
import json
import base64
import asyncio
import threading

import numpy as np

try:
    from scipy.signal import stft as _stft
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _stft = None

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import sounddevice as sd
except Exception:  # playback is optional; the dashboard still works without it
    sd = None

from aiohttp import web as aioweb, WSMsgType


OUTPUT_RATE = 16000
WAVE_POINTS = 480          # decimated waveform envelope length sent to the UI
SPEC_MAX_F = 128           # spectrogram frequency bins sent to the UI
SPEC_MAX_T = 160           # spectrogram time columns sent to the UI
SPEC_DB_RANGE = 70.0       # dynamic range (dB below per-frame peak) shown
RAW_LIVE_SECONDS = 3.0     # single window: live panels, freeze snapshot, replay


# ---------------------------------------------------------------------------
# Small DSP helpers — turn a raw signal into something cheap to ship + draw.
# ---------------------------------------------------------------------------
def wave_envelope(x, n=WAVE_POINTS):
    """Peak-envelope decimation: one max-abs value per bucket, normalized to
    [0, 1]. Cheap to compute, and draws a faithful waveform silhouette."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return [0.0] * n
    if x.size <= n:
        env = np.abs(x)
        env = np.pad(env, (0, n - env.size))
    else:
        # Trim to a multiple of n, reshape into n buckets, take max-abs each.
        usable = (x.size // n) * n
        env = np.abs(x[:usable]).reshape(n, -1).max(axis=1)
    peak = float(env.max())
    if peak > 1e-9:
        env = env / peak
    return [round(float(v), 4) for v in env]


def _bin_rows(m, target):
    """Average-pool axis 0 of `m` down to `target` rows (no-op if already <=)."""
    r = m.shape[0]
    if r <= target:
        return m
    idx = (np.arange(r) * target // r)
    out = np.zeros((target, m.shape[1]), dtype=m.dtype)
    cnt = np.zeros(target, dtype=np.int32)
    for i, b in enumerate(idx):
        out[b] += m[i]
        cnt[b] += 1
    cnt[cnt == 0] = 1
    return out / cnt[:, None]


def _bin_cols(m, target):
    """Average-pool axis 1 of `m` down to `target` columns (no-op if <=)."""
    return _bin_rows(m.T, target).T


def spectrogram(x, sr=OUTPUT_RATE):
    """Compute a small log-magnitude spectrogram as a uint8 image the browser
    can paint directly. Returns {w, h, data(row-major, freq desc), fmax}."""
    x = np.asarray(x, dtype=np.float32)
    if _stft is None or x.size < 64:
        return {"w": 0, "h": 0, "data": [], "fmax": sr // 2}
    f, t, Z = _stft(x, fs=sr, nperseg=256, noverlap=192, boundary=None)
    mag = np.abs(Z)
    db = 20.0 * np.log10(mag + 1e-6)
    db = _bin_rows(db, SPEC_MAX_F)      # frequency bins
    db = _bin_cols(db, SPEC_MAX_T)      # time columns
    top = float(db.max())
    lo = top - SPEC_DB_RANGE
    img = np.clip((db - lo) / SPEC_DB_RANGE, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    img = img[::-1, :]                  # low freq at the bottom row
    # Ship the grid as base64 bytes, not a JSON int array: ~4x smaller on the
    # wire and far cheaper for the browser to parse (atob -> Uint8Array vs.
    # JSON.parse of 20k numbers).
    return {
        "w": int(img.shape[1]),
        "h": int(img.shape[0]),
        "b64": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode("ascii"),
        "fmax": int(sr // 2),
    }


def parse_multiarray(msg):
    """Split a dorai Float32MultiArray into (header, samples[T, M]).

    Header = [capture_s, sample_rate, num_channels, seq]; samples follow
    interleaved [ch0_s0, ch1_s0, ..., ch0_s1, ...]."""
    data = np.asarray(msg.data, dtype=np.float32)
    off = msg.layout.data_offset or 4
    if data.size < off:
        return None, None
    header = data[:off]
    m = max(1, int(header[2]) if off >= 3 else 1)
    body = data[off:]
    usable = (body.size // m) * m
    frames = body[:usable].reshape(-1, m)     # [T, M]
    return header, frames


# ---------------------------------------------------------------------------
# ROS node: subscribe to the pipeline, summarize, and push to the web layer.
# ---------------------------------------------------------------------------
class WebMod(Node):
    def __init__(self, hub):
        super().__init__("web_mod")
        self.hub = hub

        self.declare_parameter("raw_topic", "/dorai_raw_audio")
        self.declare_parameter("raw_fast_topic", "/dorai_raw_fast")
        self.declare_parameter("clean_topic", "/dorai_clean_audio")
        self.declare_parameter("transcript_topic", "/dorai_transcript")
        self.declare_parameter("partial_topic", "/dorai_partial_transcript")
        self.declare_parameter("diag_topic", "/voice_mod/diagnostics")
        self.declare_parameter("levels_topic", "/voice_mod/levels")

        gp = self.get_parameter
        self.create_subscription(
            Float32MultiArray, gp("raw_topic").value, self.on_raw, 10)
        self.create_subscription(
            Float32MultiArray, gp("raw_fast_topic").value, self.on_raw_fast, 10)
        self.create_subscription(
            Float32MultiArray, gp("clean_topic").value, self.on_clean, 10)
        self.create_subscription(
            String, gp("transcript_topic").value, self.on_transcript, 10)
        self.create_subscription(
            String, gp("partial_topic").value, self.on_partial, 10)
        self.create_subscription(
            String, gp("diag_topic").value, self.on_diag, 10)
        self.create_subscription(
            Float32MultiArray, gp("levels_topic").value, self.on_levels, 10)

        self.get_logger().info("web_mod subscribed; dashboard bridge ready.")

    # ------------------------------------------------------------- callbacks
    # All rolling windows live in the Hub so freeze/live manage one consistent
    # length. When frozen we stop appending and stop pushing visuals so the
    # on-screen window is exactly what gets replayed.
    def on_raw(self, msg):
        # Full-rate multichannel frame: feeds the per-mic buffer used only for
        # per-mic playback. The visible "before" panel is the fast tap below.
        if self.hub.frozen:
            return
        _, frames = parse_multiarray(msg)
        if frames is None or frames.size == 0:
            return
        self.hub.push_raw([frames[:, i].copy() for i in range(frames.shape[1])])

    def on_raw_fast(self, msg):
        # Fast mono tap -> continuously scrolling "before" spectrogram.
        if self.hub.frozen:
            return
        _, frames = parse_multiarray(msg)
        if frames is None or frames.size == 0:
            return
        win = self.hub.append_raw_live(frames[:, 0])
        self.hub.broadcast({
            "type": "rawlive",
            "dur": round(win.size / OUTPUT_RATE, 2),
            "wave": wave_envelope(win),
            "spec": spectrogram(win),
        })

    def on_clean(self, msg):
        if self.hub.frozen:
            return
        header, frames = parse_multiarray(msg)
        if frames is None or frames.size == 0:
            return
        win = self.hub.append_clean_live(frames[:, 0].copy())
        self.hub.broadcast({
            "type": "clean",
            "seq": int(header[3]) if header.size >= 4 else 0,
            "dur": round(win.size / OUTPUT_RATE, 2),
            "wave": wave_envelope(win),
            "spec": spectrogram(win),
        })

    def on_transcript(self, msg):
        if self.hub.frozen:                          # pause transcript with capture
            return
        self.hub.broadcast({"type": "transcript", "text": msg.data, "final": True})

    def on_partial(self, msg):
        if self.hub.frozen:
            return
        self.hub.broadcast({"type": "partial", "text": msg.data})

    def on_diag(self, msg):
        self.hub.broadcast({"type": "diag", "text": msg.data})

    def on_levels(self, msg):
        if self.hub.frozen:
            return
        levels = list(np.asarray(msg.data, dtype=np.float32))
        self.hub.broadcast({"type": "levels", "rms": [round(float(v), 6) for v in levels]})


# ---------------------------------------------------------------------------
# Hub: the thread-safe bridge between the ROS thread and the asyncio server.
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self, play_device=None, logger=None):
        self.loop = None
        self.clients = set()             # asyncio.Queue per websocket
        self.logger = logger
        self.play_device = play_device

        # ONE rolling window length for everything the operator sees and hears,
        # so freezing never changes the window size: the live panels, the frozen
        # snapshot, and playback all span the same RAW_LIVE_SECONDS.
        self._buf_lock = threading.Lock()
        self._win_max = int(RAW_LIVE_SECONDS * OUTPUT_RATE)
        self._raw_live = np.zeros(0, dtype=np.float32)    # displayed raw down-mix
        self._clean_live = np.zeros(0, dtype=np.float32)  # displayed clean
        self._raw_hist = []              # per-mic multichannel (per-mic playback)
        self.frozen = False              # True => live view + capture latched
        # Latched-at-freeze copies of exactly what is on screen.
        self._frozen_clean = np.zeros(0, dtype=np.float32)
        self._frozen_raw_down = np.zeros(0, dtype=np.float32)
        self._frozen_raw = []

    # ---- rolling windows (appended by ROS thread while live) ---------------
    def append_raw_live(self, x):
        with self._buf_lock:
            self._raw_live = np.concatenate((self._raw_live, x))[-self._win_max:]
            return self._raw_live.copy()

    def append_clean_live(self, clean):
        with self._buf_lock:
            self._clean_live = np.concatenate(
                (self._clean_live, clean))[-self._win_max:]
            return self._clean_live.copy()

    def push_raw(self, chans):
        with self._buf_lock:
            if len(self._raw_hist) != len(chans):
                self._raw_hist = [c[-self._win_max:].copy() for c in chans]
            else:
                for i, c in enumerate(chans):
                    self._raw_hist[i] = np.concatenate(
                        (self._raw_hist[i], c))[-self._win_max:]

    def get_play_buffer(self, which, mic):
        with self._buf_lock:
            if which == "clean":
                b = self._frozen_clean if self.frozen else self._clean_live
                return b.copy() if b.size else None
            if mic is None or mic < 0:     # down-mix == exactly the shown panel
                b = self._frozen_raw_down if self.frozen else self._raw_live
                return b.copy() if b.size else None
            raw = self._frozen_raw if self.frozen else self._raw_hist
            if 0 <= mic < len(raw):
                return raw[mic].copy()
            return None

    # ---- freeze / resume the live pipeline view ----------------------------
    def freeze(self):
        """Latch exactly what is on screen (the live windows) for display AND
        playback, so freezing changes neither the window size nor its content."""
        self.frozen = True                       # stop further appends first
        with self._buf_lock:
            clean = self._clean_live.copy()
            rawd = self._raw_live.copy()
            n = rawd.size
            raw = [(c[-n:].copy() if n and c.size >= n else c.copy())
                   for c in self._raw_hist]
            self._frozen_clean = clean
            self._frozen_raw_down = rawd
            self._frozen_raw = raw
        dur = round(clean.size / OUTPUT_RATE, 2)
        if clean.size:
            self.broadcast({
                "type": "clean", "seq": -1, "dur": dur,
                "wave": wave_envelope(clean), "spec": spectrogram(clean),
            })
        if rawd.size:
            self.broadcast({
                "type": "raw", "seq": -1, "mics": len(raw),
                "dur": round(rawd.size / OUTPUT_RATE, 2),
                "waves": [wave_envelope(rawd)],       # down-mix, matches live
                "spec": spectrogram(rawd),
            })
        self.broadcast({"type": "state", "frozen": True, "dur": dur})
        return dur

    def live(self):
        # Fresh window on Start: clear every buffer so the next Freeze latches
        # only audio captured since this Start.
        with self._buf_lock:
            self._raw_live = np.zeros(0, dtype=np.float32)
            self._clean_live = np.zeros(0, dtype=np.float32)
            self._raw_hist = []
            self._frozen_clean = np.zeros(0, dtype=np.float32)
            self._frozen_raw_down = np.zeros(0, dtype=np.float32)
            self._frozen_raw = []
        self.frozen = False
        self.broadcast({"type": "state", "frozen": False})

    def _output_rate(self):
        """Native rate of the chosen output device. The pipeline runs at 16 kHz
        but many USB/HDA cards only open their output at 44.1/48 kHz, so we
        resample the frame to the device rate before playing (opening the stream
        at 16 kHz raises PaErrorCode -9997 'Invalid sample rate')."""
        try:
            dev = self.play_device
            if dev is None:
                dev = sd.default.device[1]        # default output index
            info = sd.query_devices(dev, "output")
            return int(info["default_samplerate"])
        except Exception:
            return OUTPUT_RATE

    def play(self, which, mic=None):
        if sd is None:
            return {"status": "no-audio-backend", "dur": 0.0}
        buf = self.get_play_buffer(which, mic)
        if buf is None or buf.size == 0:
            return {"status": "no-frame-yet", "dur": 0.0}
        dur = round(buf.size / OUTPUT_RATE, 2)     # playback length (16 kHz base)
        try:
            buf = buf.astype(np.float32)
            peak = float(np.max(np.abs(buf)))
            if peak > 1e-6:
                buf = (buf / peak) * 0.9         # normalize so raw is audible
            rate = self._output_rate()
            if rate != OUTPUT_RATE:
                import math
                from scipy.signal import resample_poly
                g = math.gcd(rate, OUTPUT_RATE)
                buf = resample_poly(buf, rate // g, OUTPUT_RATE // g).astype(np.float32)
            sd.stop()
            sd.play(buf, rate, device=self.play_device)
            return {"status": "playing", "dur": dur}
        except Exception as e:
            if self.logger:
                self.logger.error(f"playback failed: {e}")
            return {"status": f"error:{e}", "dur": 0.0}

    def stop(self):
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass

    # ---- fan-out to browsers (called from the ROS thread) ------------------
    def broadcast(self, msg):
        if self.loop is None or not self.clients:
            return
        for q in list(self.clients):
            self.loop.call_soon_threadsafe(self._safe_put, q, msg)

    @staticmethod
    def _safe_put(q, msg):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()          # drop oldest, keep latest
                q.put_nowait(msg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# aiohttp handlers.
# ---------------------------------------------------------------------------
def make_app(hub, static_dir):
    app = aioweb.Application()
    index_path = os.path.join(static_dir, "index.html")

    async def index(_req):
        return aioweb.FileResponse(index_path)

    async def ws_handler(req):
        ws = aioweb.WebSocketResponse(max_msg_size=0)
        await ws.prepare(req)
        q = asyncio.Queue(maxsize=32)
        hub.clients.add(q)

        async def sender():
            while True:
                msg = await q.get()
                await ws.send_str(json.dumps(msg))

        send_task = asyncio.ensure_future(sender())
        # Tell the freshly-connected client the current pipeline state so its
        # Start/Stop buttons render correctly.
        await ws.send_str(json.dumps({"type": "state", "frozen": hub.frozen}))
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(m.data)
                except Exception:
                    continue
                action = cmd.get("cmd")
                if action == "play":
                    which = cmd.get("which", "clean")
                    res = hub.play(which, cmd.get("mic"))
                    await ws.send_str(json.dumps({
                        "type": "play", "which": which,
                        "status": res["status"], "dur": res["dur"]}))
                elif action == "stop":
                    hub.stop()
                    await ws.send_str(json.dumps({"type": "play", "status": "stopped"}))
                elif action == "freeze":
                    dur = hub.freeze()
                    await ws.send_str(json.dumps(
                        {"type": "ack", "status": f"frozen {dur:.1f}s"}))
                elif action == "live":
                    hub.live()
                    await ws.send_str(json.dumps({"type": "ack", "status": "live"}))
        finally:
            send_task.cancel()
            hub.clients.discard(q)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    return app


def _resolve_static_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory("web_mod"), "static")
        if os.path.exists(os.path.join(d, "index.html")):
            return d
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------------------
# Entry point: run rclpy on a thread, aiohttp on the main loop.
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    # Bootstrap a temp node just to read server params (host/port/device).
    boot = Node("web_mod_boot")
    boot.declare_parameter("host", "0.0.0.0")
    boot.declare_parameter("port", 8080)
    boot.declare_parameter("play_device", "")     # sounddevice name substring
    host = boot.get_parameter("host").value
    port = int(boot.get_parameter("port").value)
    play_device = str(boot.get_parameter("play_device").value or "").strip() or None
    boot.destroy_node()

    hub = Hub(play_device=play_device)
    node = WebMod(hub)
    hub.logger = node.get_logger()

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), name="web-ros", daemon=True)
    ros_thread.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    hub.loop = loop

    app = make_app(hub, _resolve_static_dir())
    node.get_logger().info(f"dorai dashboard on http://{host}:{port}  (play_device={play_device})")

    runner = aioweb.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = aioweb.TCPSite(runner, host, port)
    loop.run_until_complete(site.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
