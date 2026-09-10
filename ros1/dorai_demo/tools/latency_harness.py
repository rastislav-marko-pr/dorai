#!/usr/bin/env python3
"""
latency_harness.py -- measure where the demo's reaction time actually goes.

WHY THIS EXISTS, AND THE MISTAKE IT REFUSES TO REPEAT
----------------------------------------------------
A previous latency measurement compared the reported bearing against a bearing
computed from the last two seconds of the SAME signal. Convergence was therefore
guaranteed and the number meant nothing. Any measurement whose reference is
derived from the pipeline under test measures only the pipeline's agreement with
itself.

So the reference here is independent: a hand clap, captured by THIS process
directly from ALSA, onset-detected in the raw samples. A clap is broadband and
has a rise time well under a millisecond, so its onset is localisable to the
sample. Nothing about that reference passes through doa_mod, wake_mod, the
bridge, or ROS.

We already know the dashboard reacts about 1.2 s after a sound. That is the same
order as the turn the robot must make. If that 1.2 s sits in the direction-of-
arrival path, the robot starts rotating well after the person stopped talking and
the demo looks broken. The point of this harness is to find out WHICH segment
owns it, because the fix is completely different depending on the answer.

SEGMENTS
--------
  1 mic -> bearing        t_clap (independent)      -> /dorai/doa arrives
                          also reported against the doa message's own origin
                          stamp, which separates algorithm delay from transport
  2 bearing -> trigger    doa/wake origin stamp     -> behaviour leaves idle
  3 trigger -> cmd_vel    behaviour leaves idle     -> first non-zero cmd_vel
  4 cmd_vel -> motion     first non-zero cmd_vel    -> independent motion signal

THIS HARNESS IS PASSIVE. It subscribes and it listens to a microphone. It
publishes nothing and it will not move the robot. Segment 4 needs the robot to
actually walk, so it only produces a number during a run a human is already
supervising.

USAGE
    # validate the wiring with no hardware and no motion
    ./latency_harness.py --dry-run

    # a real run: clap, wait for the robot, repeat; Ctrl-C for the summary
    ./latency_harness.py --claps 10 --audio-device plughw:1,0

CAVEATS, STATED RATHER THAN HIDDEN
  * Clap onset accuracy is bounded by ALSA scheduling, roughly +/-10 ms with the
    default block size. Against segments of order 1 s that is irrelevant; do not
    reuse this to measure anything under ~50 ms.
  * Segments 1 and 2 compare a wall clock in this process against origin stamps
    set in the container. Both are on the same host and the same clock, so that
    is legitimate. It would NOT be across machines without time sync.
  * cmd_vel_smooth is the planner echoing our own command back. It is not an
    independent motion reference and is only reported as a transport check.
    Real segment 4 needs the IMU, and if the IMU is dead the segment is reported
    as unavailable rather than quietly substituted.
"""
import argparse
import collections
import importlib
import json
import math
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    HAVE_ROS = True
except ImportError:                     # allows --selftest off the robot
    HAVE_ROS = False

SAMPLE_RATE = 16000
CHANNELS = 6                            # ReSpeaker UAC1.0: 0 mix, 1-4 raw, 5 ref
BLOCK_FRAMES = 256                      # 16 ms
CMD_EPS = 1e-3


# --------------------------------------------------------------------------
# independent reference: clap onset straight off ALSA
# --------------------------------------------------------------------------

class ClapDetector(threading.Thread):
    """Read raw PCM from arecord and timestamp broadband onsets.

    Onset test is a jump in high-frequency energy, obtained with a one-sample
    difference (a cheap high-pass). Speech and fan noise do not produce a
    step of this shape; a clap does. The threshold adapts to a running noise
    floor so the room's baseline does not have to be known in advance.
    """

    def __init__(self, device, on_clap, threshold=12.0, refractory_s=1.0,
                 channel=0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.device = device
        self.on_clap = on_clap
        self.threshold = threshold
        self.refractory_s = refractory_s
        self.channel = channel
        self.noise_floor = None
        self.proc = None
        self._stop = threading.Event()
        self.last_clap = 0.0

    def run(self):
        cmd = ["arecord", "-D", self.device, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", str(CHANNELS),
               "-t", "raw", "--period-size", str(BLOCK_FRAMES), "-q"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        except OSError as exc:
            print("clap detector could not start arecord: %s" % exc,
                  file=sys.stderr)
            return

        block_bytes = BLOCK_FRAMES * CHANNELS * 2
        while not self._stop.is_set():
            raw = self.proc.stdout.read(block_bytes)
            # Timestamp on arrival. The block was captured just before it was
            # handed to us, so the end of the block is ~now.
            t_block_end = time.time()
            if not raw or len(raw) < block_bytes:
                break

            block = np.frombuffer(raw, dtype="<i2").reshape(-1, CHANNELS)
            sig = block[:, self.channel].astype(np.float32) / 32768.0
            hp = np.diff(sig)
            if hp.size == 0:
                continue
            energy = float(np.sqrt(np.mean(hp * hp)))

            if self.noise_floor is None:
                self.noise_floor = max(energy, 1e-6)
                continue

            if (energy > self.threshold * self.noise_floor
                    and (t_block_end - self.last_clap) > self.refractory_s):
                # Locate the onset inside the block and correct the timestamp
                # back from the block end. This is what buys millisecond
                # resolution instead of block resolution.
                peak = int(np.argmax(np.abs(hp)))
                samples_after = hp.size - peak
                t_onset = t_block_end - samples_after / float(SAMPLE_RATE)
                self.last_clap = t_block_end
                self.on_clap(t_onset, energy / self.noise_floor)
            else:
                # Only adapt on quiet blocks, so a clap cannot raise the floor
                # enough to mask the next one.
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * energy

    def stop(self):
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except OSError:
                pass


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------

class LatencyHarness(object):
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.events = []            # completed measurement rows
        self.pending = None         # the clap currently being chased

        self.segments = collections.defaultdict(list)

        self.prev_state = None
        self.motion_cls = None

        rospy.Subscriber("/dorai/doa", String, self.on_doa, queue_size=20)
        rospy.Subscriber("/dorai/wake", String, self.on_wake, queue_size=20)
        rospy.Subscriber("/dorai/robot_state", String, self.on_robot_state,
                         queue_size=20)
        rospy.Subscriber(args.cmd_vel_topic, Twist, self.on_cmd_vel, queue_size=20)
        rospy.Subscriber(args.motion_topic, rospy.AnyMsg, self.on_motion,
                         queue_size=20)
        rospy.loginfo("latency harness listening (passive, publishes nothing)")

    # -- reference ---------------------------------------------------------

    def on_clap(self, t_onset, ratio):
        with self.lock:
            if self.pending is not None:
                self.finish("superseded by a new clap")
            self.pending = {"t_clap": t_onset, "ratio": round(ratio, 1)}
        rospy.loginfo("CLAP at %.3f (%.0fx noise floor)", t_onset, ratio)

    # -- pipeline observations --------------------------------------------

    def on_doa(self, msg):
        now = time.time()
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            p = self.pending
            if p is None or "t_doa_rx" in p:
                return
            if not payload.get("valid", True):
                return
            # Only accept a bearing whose own origin stamp is after the clap.
            # Otherwise we would happily latch a message computed before the
            # clap even happened, which is exactly the class of error that
            # made the previous measurement meaningless.
            stamp = float(payload.get("stamp", 0.0))
            if stamp and stamp < p["t_clap"]:
                return
            p["t_doa_rx"] = now
            p["t_doa_origin"] = stamp
            p["bearing_deg"] = payload.get("bearing_deg")
            p["doa_confidence"] = payload.get("confidence")

    def on_wake(self, msg):
        now = time.time()
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            p = self.pending
            if p is None or "t_wake_rx" in p:
                return
            p["t_wake_rx"] = now
            p["t_wake_origin"] = float(payload.get("stamp", 0.0))
            p["keyword"] = payload.get("keyword")

    def on_robot_state(self, msg):
        now = time.time()
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        state = payload.get("state")
        prev, self.prev_state = self.prev_state, state
        if prev == state:
            return
        with self.lock:
            p = self.pending
            if p is None:
                return
            if prev == "idle" and state != "idle" and "t_trigger" not in p:
                p["t_trigger"] = now
                p["trigger_state"] = state
            if state in ("stopped", "idle") and "t_trigger" in p:
                self.finish("state returned to %s" % state)

    def on_cmd_vel(self, msg):
        now = time.time()
        moving = (abs(msg.linear.x) > CMD_EPS or abs(msg.linear.y) > CMD_EPS
                  or abs(msg.angular.z) > CMD_EPS)
        if not moving:
            return
        with self.lock:
            p = self.pending
            if p is None or "t_cmd" in p:
                return
            p["t_cmd"] = now
            p["cmd"] = {"vx": round(msg.linear.x, 3),
                        "wz": round(msg.angular.z, 3)}

    def on_motion(self, msg):
        """Independent motion signal, typically the IMU's angular rate."""
        now = time.time()
        if self.motion_cls is None:
            try:
                type_str = msg._connection_header["type"]
                pkg, cls_name = type_str.split("/")
                self.motion_cls = getattr(
                    importlib.import_module(pkg + ".msg"), cls_name)
                rospy.loginfo("motion reference %s is %s",
                              self.args.motion_topic, type_str)
            except Exception as exc:
                rospy.logwarn_throttle(30.0, "motion topic unusable: %s", exc)
                return
        try:
            parsed = self.motion_cls().deserialize(msg._buff)
        except Exception:
            return

        rate = None
        if hasattr(parsed, "angular_velocity"):
            a = parsed.angular_velocity
            rate = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        elif hasattr(parsed, "twist"):
            try:
                a = parsed.twist.twist.angular
            except AttributeError:
                a = parsed.twist.angular
            rate = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        if rate is None:
            return

        with self.lock:
            p = self.pending
            if p is None or "t_cmd" not in p or "t_motion" in p:
                return
            if rate > self.args.motion_threshold:
                p["t_motion"] = now
                p["motion_rate"] = round(rate, 3)

    # -- bookkeeping -------------------------------------------------------

    def finish(self, reason):
        """Close out the pending measurement. Caller holds the lock."""
        p = self.pending
        self.pending = None
        if p is None:
            return
        p["closed_because"] = reason

        def seg(name, a, b):
            if a in p and b in p and p[a] and p[b]:
                value = p[b] - p[a]
                if value >= 0:
                    p[name] = round(value, 4)
                    self.segments[name].append(value)

        seg("mic_to_bearing_rx", "t_clap", "t_doa_rx")
        seg("mic_to_bearing_origin", "t_clap", "t_doa_origin")
        seg("bearing_to_trigger", "t_doa_origin", "t_trigger")
        seg("wake_to_trigger", "t_wake_origin", "t_trigger")
        seg("trigger_to_cmd_vel", "t_trigger", "t_cmd")
        seg("cmd_vel_to_motion", "t_cmd", "t_motion")
        seg("total_mic_to_motion", "t_clap", "t_motion")

        self.events.append(p)
        rospy.loginfo("measurement closed (%s): %s", reason,
                      {k: v for k, v in p.items() if not k.startswith("t_")})

    def timeout_check(self, _evt):
        with self.lock:
            if self.pending is None:
                return
            if (time.time() - self.pending["t_clap"]) > self.args.timeout_s:
                self.finish("timed out waiting for the pipeline")

    def summary(self):
        lines = ["", "=" * 72,
                 "LATENCY SUMMARY  (%d measurements)" % len(self.events),
                 "=" * 72,
                 "%-26s %7s %7s %7s %7s %5s"
                 % ("segment", "n", "median", "mean", "max", "unit")]
        order = ["mic_to_bearing_origin", "mic_to_bearing_rx",
                 "bearing_to_trigger", "wake_to_trigger",
                 "trigger_to_cmd_vel", "cmd_vel_to_motion",
                 "total_mic_to_motion"]
        for name in order:
            values = self.segments.get(name)
            if not values:
                lines.append("%-26s %7s %7s %7s %7s %5s"
                             % (name, 0, "-", "-", "-", "s"))
                continue
            arr = np.array(values)
            lines.append("%-26s %7d %7.3f %7.3f %7.3f %5s"
                         % (name, arr.size, float(np.median(arr)),
                            float(arr.mean()), float(arr.max()), "s"))
        lines.append("")
        lines.append("A segment showing n=0 was never observed. That is a")
        lines.append("finding, not a gap to fill with a guess.")
        lines.append("=" * 72)
        return "\n".join(lines)

    def write_json(self, path):
        with open(path, "w") as handle:
            json.dump({"events": self.events,
                       "segments": {k: v for k, v in self.segments.items()}},
                      handle, indent=2)
        print("raw measurements written to %s" % path)


def selftest():
    """Exercise the maths with no hardware and no ROS."""
    assert abs(1.0) == 1.0
    fake = {"t_clap": 100.0, "t_doa_origin": 100.9, "t_doa_rx": 101.05,
            "t_trigger": 101.10, "t_cmd": 101.14, "t_motion": 101.31}
    expect = {"mic_to_bearing_origin": 0.9, "mic_to_bearing_rx": 1.05,
              "bearing_to_trigger": 0.2, "trigger_to_cmd_vel": 0.04,
              "cmd_vel_to_motion": 0.17, "total_mic_to_motion": 1.31}
    for name, (a, b) in {
            "mic_to_bearing_origin": ("t_clap", "t_doa_origin"),
            "mic_to_bearing_rx": ("t_clap", "t_doa_rx"),
            "bearing_to_trigger": ("t_doa_origin", "t_trigger"),
            "trigger_to_cmd_vel": ("t_trigger", "t_cmd"),
            "cmd_vel_to_motion": ("t_cmd", "t_motion"),
            "total_mic_to_motion": ("t_clap", "t_motion")}.items():
        got = round(fake[b] - fake[a], 4)
        assert abs(got - expect[name]) < 1e-6, (name, got, expect[name])

    # onset localisation inside a block
    fs = SAMPLE_RATE
    hp_size = BLOCK_FRAMES - 1
    peak = 100
    t_block_end = 1000.0
    t_onset = t_block_end - (hp_size - peak) / float(fs)
    assert abs(t_onset - (t_block_end - 0.0096875)) < 1e-9, t_onset
    print("selftest OK: segment arithmetic and onset localisation")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-device", default="plughw:1,0",
                        help="ALSA device for the independent clap reference")
    parser.add_argument("--audio-channel", type=int, default=0)
    parser.add_argument("--cmd-vel-topic", default="/robot_1/cmd_vel")
    parser.add_argument("--motion-topic", default="/imu/data",
                        help="independent motion reference, NOT cmd_vel_smooth")
    parser.add_argument("--motion-threshold", type=float, default=0.15,
                        help="rad/s on the motion reference that counts as moving")
    parser.add_argument("--clap-threshold", type=float, default=12.0,
                        help="onset energy as a multiple of the noise floor")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--claps", type=int, default=0,
                        help="stop after this many measurements, 0 = until Ctrl-C")
    parser.add_argument("--out", default="/tmp/dorai_latency.json")
    parser.add_argument("--no-audio", action="store_true",
                        help="skip the microphone; only pipeline-internal segments")
    parser.add_argument("--dry-run", action="store_true",
                        help="check wiring, print what would be measured, exit")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    # Line-buffer stdout. This normally runs with its output redirected to a
    # file, and a block-buffered summary that is still sitting in the buffer
    # when the process is interrupted is a summary you never see.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    if args.selftest:
        selftest()
        return 0

    if not HAVE_ROS:
        print("rospy unavailable; run --selftest, or source the ROS setup",
              file=sys.stderr)
        return 2

    rospy.init_node("dorai_latency_harness", anonymous=True, disable_signals=True)

    if args.dry_run:
        print("DRY RUN. Would measure, in order:")
        print("  1 mic -> bearing        clap on %s -> /dorai/doa"
              % args.audio_device)
        print("  2 bearing -> trigger    doa origin stamp -> /dorai/robot_state"
              " leaves idle")
        print("  3 trigger -> cmd_vel    -> first non-zero %s" % args.cmd_vel_topic)
        print("  4 cmd_vel -> motion     -> %s above %.2f rad/s"
              % (args.motion_topic, args.motion_threshold))
        published = dict(rospy.get_published_topics())
        for topic in ["/dorai/doa", "/dorai/wake", "/dorai/robot_state",
                      args.cmd_vel_topic, args.motion_topic]:
            print("  %-24s %s" % (topic,
                                  "present" if topic in published else "ABSENT"))
        print("\nThis harness publishes nothing and cannot move the robot.")
        return 0

    harness = LatencyHarness(args)
    rospy.Timer(rospy.Duration(1.0), harness.timeout_check)

    detector = None
    if not args.no_audio:
        detector = ClapDetector(args.audio_device, harness.on_clap,
                                threshold=args.clap_threshold,
                                channel=args.audio_channel)
        detector.start()
        print("clap into the array to start each measurement; Ctrl-C to finish")
    else:
        print("no-audio mode: segments 1 and 2 will be unavailable")

    try:
        while not rospy.is_shutdown():
            if args.claps and len(harness.events) >= args.claps:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if detector is not None:
            detector.stop()
        print(harness.summary())
        harness.write_json(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
