#!/usr/bin/env python3
"""
doa_mod / doa_node.py — direction of arrival on /dorai/doa.

    /dorai_raw_mc  (4-ch raw ReSpeaker capsules, 16 kHz)
        -> neural VAD gate
        -> per-frame least-squares TDOA (closure check only, NOT the bearing)
        -> pooled SRP-PHAT accumulator  <- THE bearing
        -> /dorai/doa   std_msgs/String carrying the DEMO_INTERFACES JSON

WHAT THIS NODE IS AND IS NOT
----------------------------
It is a ROS 2 wrapper. Every number it reports comes from `doa_core.py`,
which is vendored here byte-identical to `updated_beamformer/doa/doa_core.py`
and must stay that way: it carries three sign/geometry bug fixes and a
21-test regression suite (`tests/test_doa_core.py`, run it before believing
anything this node says). The estimator path is lifted from the vendor's
`live_doa.py::DoaApp.update()` with the matplotlib half deleted and nothing
else changed.

Four things this node is careful to get right, each of which the vendor paid
for in measured error on real recordings:

1.  THE BEARING IS THE POOLED SRP-PHAT PEAK, not a circular average of
    per-frame least-squares bearings. The per-frame fit is still computed,
    but only as a diagnostic: it drives the closure gate. Pooling raw power
    and taking the argmax once took mean absolute error from 48.4 deg to
    13.0 deg on the 2026-08-04 session. See doa_core.SRPAccumulator.

2.  RAW POWER IS ACCUMULATED, normalised once at the end. Per-frame min-max
    normalisation equalises frames of wildly different SNR and moved the
    pooled peak by 4-8 deg. SRPAccumulator already does this; we do not
    touch the pool.

3.  TDOA CLOSURE CARRIES THROUGH AS `valid`. Closure (tau_ik - tau_ij -
    tau_jk == 0) holds for any single source regardless of geometry, so
    unlike a plane-wave residual it separates "bad frame" from "bad
    geometry". Per DEMO_INTERFACES, `valid` is false when closure fails on
    more than `closure_fail_frac` of recent speech frames, or when there is
    no speech at all -- and `bearing_deg` is still published, just not to be
    trusted. `confidence` degrades smoothly with the closure pass rate, so a
    marginal room reads as low confidence rather than as a confident wrong
    number.

4.  STATIONARY-INTERFERER SUBTRACTION is available and on by default
    (`subtract_noise`). It accumulates a second spectrum over NON-speech
    frames and subtracts it. On the vendor's session a fan at ~280 deg
    dominated every take; this recovered 13.0 -> 7.5 deg. It assumes the
    interferer really is stationary -- it is a mitigation, not a substitute
    for switching the fan off.

WHY IT SUBSCRIBES INSTEAD OF OPENING THE MIC
--------------------------------------------
The ReSpeaker 4 Mic Array is a single UAC1.0 device and therefore a single
exclusive ALSA capture endpoint. voice_mod owns it. A second open fails, or
worse, succeeds and starves the first. So doa_mod takes voice_mod's raw
multichannel tap, which is the same audio, on the same clock, before any
beamforming. `/dorai_raw_mc` exists for exactly this (see voice_mod's
`publish_raw_mc`): `/dorai_raw_audio` is multichannel but only arrives once
per publish_interval, and `/dorai_raw_fast` is fast but mono-downmixed, which
annihilates the inter-channel delays a bearing is made of.

TIMESTAMPS
----------
The tap's 4-float header carries `capture_s`, seconds since *voice_mod's*
start, not an epoch -- a float32 cannot hold an epoch to better than 128 s.
So this node estimates the offset between the two clocks as the MINIMUM of
(receive epoch - capture_s) over a sliding window: the minimum is the sample
that suffered the least transport delay, which makes it a stable, low-bias
estimate of voice_mod's t0 in epoch terms. The published `stamp` is the
midpoint of the analysis block mapped through that offset, i.e. the acoustic
instant the bearing describes, as DEMO_INTERFACES requires ("taken at the
origin of the data, not at send time").
"""

import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doa_core as dc                                        # noqa: E402

try:
    from vad import NeuralVAD as _NeuralVAD
    _HAS_VAD = True
except Exception:                                            # pragma: no cover
    _HAS_VAD = False


SCHEMA_V = 1


class DoaMod(Node):
    def __init__(self):
        super().__init__("doa_mod")
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        # --- plumbing ------------------------------------------------------
        self.declare_parameter("input_topic", "/dorai_raw_mc")
        self.declare_parameter("output_topic", "/dorai/doa")
        # Comma-separated indices INTO THE INCOMING TAP, when the tap carries
        # more than the four capsules. Empty = use every channel it sends,
        # which is the normal case: voice_mod is already configured with
        # array_channels=1,2,3,4 so the tap is exactly ch1..ch4 in order.
        self.declare_parameter("channels", "")

        # --- geometry: the KNOWN-CIRCLE shortcut ---------------------------
        # 32 mm is the published ODAS/pyroomacoustics radius for this board,
        # and doa_core.array_positions() places MIC1 at 45 deg numbering
        # counter-clockwise, matching the Seeed drawing. We deliberately do
        # NOT run auto_calibrate here: MDS recovers geometry only up to a
        # rotation AND a reflection, so a walk-around calibration can never
        # resolve the mirror by itself, and the mirror is the failure that
        # costs 180 deg. Azimuth is invariant to radius scaling anyway (it is
        # a ratio of direction-vector components), so 32 vs the 35.7 mm the
        # real capture implies does not move the bearing.
        self.declare_parameter("radius_mm", 32.0, dyn)

        # --- framing -------------------------------------------------------
        # 2048 samples = 128 ms, the vendor's BLOCK: long enough for a stable
        # GCC-PHAT peak. hop 1600 = 100 ms = one voice_mod worker block, and
        # 10 Hz is the DEMO_INTERFACES rate cap for /dorai/doa. The vendor's
        # display hopped every 512 samples (31 Hz); at ~29 ms of SRP + LSQ per
        # frame that is most of a core for a number nobody can see change that
        # fast, so we hop at the publish rate instead.
        self.declare_parameter("block", 2048)
        self.declare_parameter("hop", 1600)
        self.declare_parameter("max_rate_hz", 10.0, dyn)

        # --- estimator -----------------------------------------------------
        # Pooled-SRP forget time constant. 2.0 s is the vendor default: long
        # enough to pool many frames, short enough to follow a talker who
        # moves.
        self.declare_parameter("avg_tc", 2.0, dyn)
        # Stationary-interferer subtraction. On by default; see the module
        # docstring. Turn it off if the interferer moves.
        self.declare_parameter("subtract_noise", True)
        # Neural VAD threshold on the reference capsule. The vendor's default.
        self.declare_parameter("vad_threshold", 0.45, dyn)
        # Where vad.onnx lives. Empty = look in the package share directory,
        # then beside vad.py. Worth being explicit about: vad.py's own default
        # is "next to this file", which lands in site-packages, and the model
        # is installed into share/ -- that only happens to work because the
        # image builds with colcon --symlink-install. Without the fallback the
        # node would silently drop to RMS energy gating, and a bearing gated on
        # energy points at the room's noise field rather than at the talker.
        self.declare_parameter("vad_model_path", "")
        # F_HI is the spatial aliasing limit of the 64 mm diagonal, NOT
        # Nyquist. Raising it does not add resolution, it adds ambiguous
        # peaks: the measured 2000-2600 Hz band already degrades to R = 0.652
        # where its neighbours hold R >= 0.99.
        self.declare_parameter("f_hi", float(dc.F_HI), dyn)
        # DEMO_INTERFACES: valid is false when closure fails on more than 70
        # percent of speech frames.
        self.declare_parameter("closure_fail_frac", 0.7, dyn)
        # Window the closure statistic is computed over. Lifetime counters (as
        # the vendor's banner used) never recover from a bad first minute.
        self.declare_parameter("closure_window_s", 3.0, dyn)

        # --- output --------------------------------------------------------
        # 36 bins, 10 deg apart, index 0 = 0 deg, per DEMO_INTERFACES. 0
        # omits the key entirely (the contract says omit rather than send
        # nulls) for hosts that cannot spare the CPU or the bytes.
        self.declare_parameter("spectrum_bins", 36)

        gp = self.get_parameter
        self.input_topic = gp("input_topic").value
        self.output_topic = gp("output_topic").value
        self.channels = [int(t) for t in
                         str(gp("channels").value or "").replace(";", ",").split(",")
                         if t.strip()]
        self.radius = float(gp("radius_mm").value) / 1000.0
        self.block = int(gp("block").value)
        self.hop = int(gp("hop").value)
        self.max_rate = float(gp("max_rate_hz").value)
        self.avg_tc = float(gp("avg_tc").value)
        self.subtract_noise = bool(gp("subtract_noise").value)
        self.vad_threshold = float(gp("vad_threshold").value)
        self.f_hi = float(gp("f_hi").value)
        self.closure_fail_frac = float(gp("closure_fail_frac").value)
        self.closure_window_s = float(gp("closure_window_s").value)
        self.spectrum_bins = int(gp("spectrum_bins").value)

        self.pos = dc.array_positions(self.radius)
        self.grid = np.arange(0.0, 360.0, 1.0)
        self.fs = dc.FS_DEFAULT          # replaced by the tap's own header
        self.acc = None                  # built on the first frame, once fs is known

        if _HAS_VAD:
            try:
                self._vad = _NeuralVAD(self._vad_model_path(
                    str(gp("vad_model_path").value or "").strip()))
                self._use_neural_vad = True
            except Exception as exc:
                self.get_logger().warning(
                    f"Neural VAD init failed ({exc}); falling back to an "
                    f"adaptive RMS percentile gate.")
                self._vad = None
                self._use_neural_vad = False
        else:
            self._vad = None
            self._use_neural_vad = False
        self._rms_hist = deque(maxlen=400)

        # Rolling multichannel buffer and its capture-time head.
        self._buf = None                 # [M, N] float64
        self._t_head = None              # capture_s of the sample past the end
        self._since_hop = 0
        # Sliding window of (epoch - capture_s). The window length is the
        # recovery time after a voice_mod restart: capture_s resets to 0, every
        # sample in here becomes 10 minutes too small, and the running minimum
        # would otherwise ratchet the stamps into the past until it aged out.
        # 100 samples at the 10 Hz tap is 10 s -- and the restart is caught
        # explicitly below, so this is only the backstop.
        self._offsets = deque(maxlen=100)   # epoch - capture_s samples
        self._last_cap0 = None
        self._closure = deque()          # (t_mono, active, ok)
        self._last_pub = 0.0
        self._last_bearing = 0.0
        self._nframes = 0
        self._npub = 0

        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.sub = self.create_subscription(
            Float32MultiArray, self.input_topic, self.on_raw, 10)
        self.create_timer(10.0, self._log_health)
        # avg_tc was declared dynamic but nothing acted on a change, so setting
        # it at runtime silently did nothing and every trial cost a restart.
        # This is the parameter people actually want to tune by feel: it is the
        # pooled-SRP forget time, so it IS the lag between a talker moving and
        # the reported bearing following them.
        self.add_on_set_parameters_callback(self._on_set_params)

        self.get_logger().info(
            f"doa_mod: {self.input_topic} -> {self.output_topic}; "
            f"circle r={self.radius * 1000:.1f}mm at "
            f"{np.rad2deg(np.arctan2(self.pos[:, 1], self.pos[:, 0])) % 360}, "
            f"block={self.block} hop={self.hop} f_hi={self.f_hi:.0f}Hz "
            f"avg_tc={self.avg_tc}s subtract_noise={self.subtract_noise} "
            f"vad={'neural' if self._use_neural_vad else 'rms'}")

    def _on_set_params(self, params):
        """Apply a live avg_tc change without dropping accumulated evidence.

        The accumulator stores the forget factor, not the time constant:
        alpha = exp(-hop / (tc * fs)). Recomputing alpha in place retunes the
        averaging while KEEPING the spatial spectrum built up so far, which is
        what you want mid-session. Rebuilding the accumulator instead would
        throw away the noise estimate that --subtract-noise depends on, and that
        estimate is worth 5 degrees of accuracy on a room with a steady
        interferer.

        doa_core.py is vendored byte-identical to the vendor's release and is
        checksum-checked, so the formula is duplicated here rather than adding a
        setter over there. If the vendor ever changes the decay convention this
        is the line that has to follow it.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name != "avg_tc":
                continue
            tc = float(p.value)
            if not (tc > 0.0):
                return SetParametersResult(
                    successful=False, reason="avg_tc must be greater than zero")
            self.avg_tc = tc
            if getattr(self, "acc", None) is not None:
                self.acc.alpha = float(np.exp(-self.hop / (tc * self.acc.fs)))
                self.get_logger().info(
                    f"avg_tc -> {tc:.2f}s (alpha {self.acc.alpha:.4f}); "
                    f"lower reacts faster to a moving talker, higher is steadier")
        return SetParametersResult(successful=True)

    @staticmethod
    def _vad_model_path(explicit):
        """vad.onnx, wherever it ended up. None lets vad.py use its own default."""
        cands = [explicit] if explicit else []
        try:
            from ament_index_python.packages import get_package_share_directory
            cands.append(os.path.join(
                get_package_share_directory("doa_mod"), "vad.onnx"))
        except Exception:
            pass
        cands.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vad.onnx"))
        for c in cands:
            if c and os.path.isfile(c):
                return c
        return None

    # -- input ---------------------------------------------------------------
    def on_raw(self, msg):
        """De-interleave one raw multichannel chunk into the rolling buffer.

        Header contract (shared with /dorai_raw_audio, see the project README):
        data[0:4] = [capture_s, sample_rate, num_channels, sequence_number],
        then interleaved samples.
        """
        recv = time.time()
        data = msg.data
        if len(data) < 5:
            return
        cap0 = float(data[0])
        fs = int(data[1])
        m = int(data[2])
        if m < 2:
            self.get_logger().error(
                f"{self.input_topic} carries {m} channel(s); a bearing needs "
                f"at least 2. Is voice_mod running with array_channels set?",
                throttle_duration_sec=10.0)
            return
        body = np.asarray(data[4:], dtype=np.float64)
        n = body.size // m
        if n <= 0:
            return
        blk = body[:n * m].reshape(n, m).T          # [M, T]
        if self.channels:
            blk = blk[self.channels, :]
            m = blk.shape[0]

        if self.acc is None:
            self._start(fs, m)
        elif fs != self.fs:
            self.get_logger().error(
                f"{self.input_topic} changed rate {self.fs} -> {fs}; ignoring.",
                throttle_duration_sec=10.0)
            return

        # Clock bridge: the tap's capture_s is voice_mod-relative. The minimum
        # of (receive epoch - capture_s) over the window is the least-delayed
        # sample and therefore the best estimate of voice_mod's t0 in epoch.
        # capture_s going BACKWARDS means voice_mod restarted and its clock
        # began again at zero, so every stored offset now describes a process
        # that no longer exists. Drop them, and drop the audio buffer with
        # them: the samples either side of a restart are not contiguous and
        # splicing them would fabricate a delay across all four channels.
        if self._last_cap0 is not None and cap0 < self._last_cap0 - 0.5:
            self.get_logger().warning(
                f"{self.input_topic} capture clock went backwards "
                f"({self._last_cap0:.1f}s -> {cap0:.1f}s): voice_mod restarted. "
                f"Resetting the clock offset and the audio buffer.")
            self._offsets.clear()
            self._buf = None
            self._since_hop = 0
        self._last_cap0 = cap0
        self._offsets.append(recv - cap0)

        if self._buf is None or self._buf.shape[0] != m:
            self._buf = np.zeros((m, 0))
        self._buf = np.concatenate((self._buf, blk), axis=1)
        self._t_head = cap0 + n / float(self.fs)
        keep = max(self.block * 4, self.block + self.hop)
        if self._buf.shape[1] > keep:
            self._buf = self._buf[:, -keep:]
        self._since_hop += n

        # One estimate per hop of NEW audio, each on its own window. `end` is
        # where this hop's window finishes, which is not the buffer end when a
        # single message carried more than one hop -- analysing the same tail
        # twice would double-count it into the pool and stall the bearing on a
        # slower tap.
        while self._since_hop >= self.hop:
            self._since_hop -= self.hop
            end = self._buf.shape[1] - self._since_hop
            if end < self.block:
                continue
            self._estimate(end)

    def _start(self, fs, m):
        self.fs = int(fs)
        if m != self.pos.shape[0]:
            # Any circular array works; the ReSpeaker's four capsules are the
            # case this was built for.
            self.pos = dc.circle_positions(m, self.radius, start_deg=45.0)
            self.get_logger().warning(
                f"tap carries {m} channels; rebuilt the circle for {m} mics.")
        self.acc = dc.SRPAccumulator(
            self.pos, fs=self.fs, grid_deg=self.grid, f_hi=self.f_hi,
            decay=self.avg_tc, hop=self.hop,
            subtract_noise=self.subtract_noise)
        self.get_logger().info(
            f"first frame: {m}ch @ {self.fs} Hz; aliasing limit "
            f"{dc.aliasing_limit_hz(self.pos):.0f} Hz, beamwidth "
            f"{dc.beamwidth_deg(self.pos, self.f_hi):.0f} deg")

    # -- one estimate --------------------------------------------------------
    def _estimate(self, end):
        block = self._buf[:, end - self.block:end]
        self._nframes += 1

        if self._use_neural_vad:
            active = self._vad.process_buffer(
                block[0], fs=self.fs, threshold=self.vad_threshold)["is_speech"]
        else:
            rms = dc.frame_rms(block)
            self._rms_hist.append(rms)
            thr = (np.percentile(self._rms_hist, 70.0)
                   if len(self._rms_hist) >= 20 else np.inf)
            active = bool(rms > thr)

        # Per-frame least squares is a DIAGNOSTIC here, never the answer: its
        # `consistent` flag is the TDOA closure test. A frame whose pairwise
        # delays cannot come from one plane wave carries no direction, and
        # averaging it in does not cancel -- it drags the pooled bearing
        # toward broadside and makes noise look like a measurement.
        est = dc.doa_least_squares(block, pos=self.pos, f_hi=self.f_hi,
                                   fs=self.fs)
        ok = bool(est.get("consistent", True))
        now = time.monotonic()
        self._closure.append((now, active, ok))
        while self._closure and now - self._closure[0][0] > self.closure_window_s:
            self._closure.popleft()

        # Feed BOTH pools: non-speech frames are what the interferer
        # subtraction is built from. Only speech frames move the bearing.
        self.acc.update(block, active=active)

        n_act = sum(1 for _, a, _ in self._closure if a)
        n_ok = sum(1 for _, a, o in self._closure if a and o)
        pass_frac = (n_ok / n_act) if n_act else 0.0
        valid = bool(n_act > 0 and (1.0 - pass_frac) <= self.closure_fail_frac)

        az, sharp = self.acc.bearing()
        if math.isnan(az):
            az = self._last_bearing
        else:
            self._last_bearing = float(az % 360.0)
        az = float(self._last_bearing)

        # Confidence degrades with the closure pass rate rather than falling
        # off a cliff at the valid/invalid boundary, so a marginal room reads
        # as "low confidence", which is what DEMO_INTERFACES asks for, instead
        # of as a confident wrong number.
        conf = float(max(0.0, min(1.0, sharp * pass_frac)))

        # Rate limit last, so the estimator state still advances every hop.
        # The window is 10 percent short of the nominal period on purpose:
        # estimates already arrive at the hop cadence, which for the default
        # hop IS the cap, and a hard 1/rate window would then discard every
        # hop that ran a millisecond early, halving the output to ~6 Hz. This
        # is a ceiling for a runaway publisher, not a clock.
        wall = time.time()
        if self.max_rate > 0 and (wall - self._last_pub) < 0.9 / self.max_rate:
            return
        self._last_pub = wall

        payload = {
            "v": SCHEMA_V,
            "topic": "doa",
            "stamp": round(self._stamp(end), 3),
            "bearing_deg": round(az, 1),
            "confidence": round(conf, 3),
            "valid": valid,
        }
        spec = self._spectrum()
        if spec is not None:
            payload["spectrum"] = spec
        self.pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._npub += 1

    def _stamp(self, end):
        """Epoch seconds of the acoustic midpoint of the analysis block."""
        if self._t_head is None or not self._offsets:
            return time.time()
        # _t_head is the capture time of the sample past the BUFFER end; walk
        # back to this window's end, then to its middle.
        t_end = self._t_head - (self._buf.shape[1] - end) / float(self.fs)
        t_mid = t_end - (self.block / 2.0) / float(self.fs)
        return t_mid + min(self._offsets)

    def _spectrum(self):
        """Pooled SRP power as `spectrum_bins` bins, index 0 centred on 0 deg.

        The pool is already min-max normalised by SRPAccumulator.spectrum();
        binning can only lower the peak, so renormalise afterwards to satisfy
        "normalised to a 0 to 1 peak". Bins are means over the 1 deg grid, not
        samples of it: with a ~130 deg main lobe a decimated sample would be
        an honest but needlessly noisy summary of the same curve.
        """
        if self.spectrum_bins <= 0 or self.acc is None or self.acc.n_speech == 0:
            return None
        p = self.acc.spectrum()
        if not np.isfinite(p).all() or p.max() <= 0:
            return None
        nb = self.spectrum_bins
        width = 360.0 / nb
        # bin k covers [k*width - width/2, k*width + width/2), centred on k*width
        b = np.floor((self.grid + width / 2.0) / width).astype(int) % nb
        out = np.zeros(nb)
        cnt = np.zeros(nb)
        np.add.at(out, b, p)
        np.add.at(cnt, b, 1.0)
        out = out / np.maximum(cnt, 1.0)
        peak = out.max()
        if peak > 0:
            out = out / peak
        return [round(float(v), 4) for v in out]

    def _log_health(self):
        if self.acc is None:
            self.get_logger().warning(
                f"no audio yet on {self.input_topic}. voice_mod needs "
                f"-p publish_raw_mc:=true.")
            return
        n_act = sum(1 for _, a, _ in self._closure if a)
        n_ok = sum(1 for _, a, o in self._closure if a and o)
        self.get_logger().info(
            f"frames={self._nframes} published={self._npub} "
            f"speech={n_act}/{len(self._closure)} in the last "
            f"{self.closure_window_s:.0f}s, closure ok={n_ok}/{max(n_act, 1)}, "
            f"bearing={self._last_bearing:.0f} deg")


def main(args=None):
    rclpy.init(args=args)
    node = DoaMod()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
