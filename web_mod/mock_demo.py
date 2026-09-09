#!/usr/bin/env python3
"""
web_mod / mock_demo.py  —  fake the demo bridge so the dashboard can be built
and tested without doa_mod, wake_mod, the UDP bridge or the robot.

It publishes exactly the four topics of DEMO_INTERFACES.md as std_msgs/String
JSON, at the contract's rates (doa 10 Hz, person 10 Hz, robot_state 5 Hz, wake
event-driven), and it plays a full demo: someone says the wake word, the robot
turns, acquires a person, walks to them, stops.

Then it deliberately breaks. The last third of the loop is the part the panels
were actually developed against, because a dashboard that only ever sees good
data is a dashboard nobody has tested:

  * `valid: false` bearings          — the needle must vanish, not park itself
  * a second acoustic source         — the polar curve must show both lobes
  * `present: false`                 — the person numbers must blank, not hold
  * a total blackout, all four       — every badge must go stale then SILENT
  * a `doa` message with no          — the panel must fall back to a bare
    `spectrum` key at all              needle instead of drawing a stale curve
  * an out-of-contract `state`       — the strip must say UNKNOWN, not guess

Usage (any ROS 2 shell that can see the dashboard's domain):

    python3 web_mod/mock_demo.py                # loop the 42 s scenario
    python3 web_mod/mock_demo.py --once         # run it once, then idle
    python3 web_mod/mock_demo.py --speed 3      # 3x faster, for impatient dev
    python3 web_mod/mock_demo.py --camera /mock/cam/compressed   # + fake video

Then, in another shell:

    ros2 run web_mod web            # or: python3 web_mod/web.py
    # with the camera:
    python3 web_mod/web.py --ros-args -p camera_topic:=/mock/cam/compressed

This is a testing tool, not part of the demo. Nothing in web.py imports it.
"""

import sys
import json
import math
import time
import argparse
import importlib

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String


SCHEMA_V = 1
CYCLE_S = 42.0            # one full scenario, including the failure cases


# ---------------------------------------------------------------------------
# Scenario helpers.
# ---------------------------------------------------------------------------
def lerp(a, b, k):
    k = max(0.0, min(1.0, k))
    return a + (b - a) * k


def srp_spectrum(lobes, floor=0.06, bins=36, t=0.0):
    """Build a plausible 36-bin SRP-PHAT curve: wrapped Gaussian lobes over a
    noise floor, peak-normalised to 1 exactly as the contract promises.

    `lobes` is a list of (bearing_deg, height, width_deg). Two entries is the
    interesting case — that is what proves the polar plot earns its space by
    revealing a source the bare bearing would hide.
    """
    out = []
    for i in range(bins):
        deg = i * 360.0 / bins
        v = floor * (0.6 + 0.4 * math.sin(t * 3.1 + i))     # a little life
        for (c, h, w) in lobes:
            d = abs(((deg - c + 180.0) % 360.0) - 180.0)    # wrapped distance
            v += h * math.exp(-0.5 * (d / w) ** 2)
        out.append(v)
    peak = max(out) or 1.0
    return [round(v / peak, 4) for v in out]


def scenario(t):
    """Map scenario time -> (doa, wake, person, robot_state) payloads.

    Any of the four may be None, which means "publish nothing this tick". That
    is how the blackout phase is expressed: silence on the wire, not a message
    saying things are bad. Silence is what a dead publisher actually looks like
    and it is the case the dashboard has to survive.
    """
    doa = wake = person = state = None

    # ---- 0-5 s: quiet room. Nobody has said anything yet. ------------------
    if t < 5.0:
        doa = dict(bearing_deg=round(200.0 + 40.0 * math.sin(t), 1),
                   confidence=round(0.05 + 0.08 * abs(math.sin(t * 0.7)), 3),
                   valid=False,
                   spectrum=srp_spectrum([], floor=0.5, t=t))
        person = dict(present=False)
        state = dict(state="idle", mode=0, cmd=dict(vx=0.0, vy=0.0, wz=0.0))

    # ---- 5-6.5 s: speech from 137 deg, then the wake word ------------------
    elif t < 6.5:
        doa = dict(bearing_deg=round(137.0 + 2.0 * math.sin(t * 4), 1),
                   confidence=round(lerp(0.4, 0.86, (t - 5.0) / 1.5), 3),
                   valid=True,
                   spectrum=srp_spectrum([(137, 1.0, 14)], t=t))
        person = dict(present=False)
        state = dict(state="idle", mode=1, cmd=dict(vx=0.0, vy=0.0, wz=0.0))

    # ---- 6.5-10 s: turn toward the wake bearing ----------------------------
    elif t < 10.0:
        doa = dict(bearing_deg=round(137.0 + 3.0 * math.sin(t * 3), 1),
                   confidence=0.81, valid=True,
                   spectrum=srp_spectrum([(137, 1.0, 13)], t=t))
        # The person appears part-way through the turn, at the edge of frame.
        if t > 8.4:
            person = dict(present=True,
                          bearing_deg=round(lerp(41.0, 18.0, (t - 8.4) / 1.6), 1),
                          distance_m=3.62, score=0.71)
        else:
            person = dict(present=False)
        state = dict(state="turn", mode=1, cmd=dict(vx=0.0, vy=0.0, wz=0.42),
                     target_bearing_deg=137.4)

    # ---- 10-13 s: acquire — line the person up before walking --------------
    elif t < 13.0:
        doa = dict(bearing_deg=round(136.0 + 4.0 * math.sin(t * 2), 1),
                   confidence=0.74, valid=True,
                   spectrum=srp_spectrum([(136, 1.0, 15)], t=t))
        person = dict(present=True,
                      bearing_deg=round(lerp(18.0, 2.5, (t - 10.0) / 3.0), 1),
                      distance_m=round(lerp(3.62, 3.44, (t - 10.0) / 3.0), 2),
                      score=round(lerp(0.71, 0.88, (t - 10.0) / 3.0), 2))
        state = dict(state="acquire", mode=1, cmd=dict(vx=0.0, vy=0.0, wz=0.18),
                     target_bearing_deg=137.4, distance_m=3.44)

    # ---- 13-24 s: approach. The speaker keeps talking and WALKS, so the -----
    #      trail on the dial gets something to draw; a second source (a
    #      colleague at 300 deg) joins at 18 s and only the curve shows it.
    elif t < 24.0:
        k = (t - 13.0) / 11.0
        bearing = lerp(137.0, 95.0, k)
        lobes = [(bearing, 1.0, 13)]
        if t > 18.0:
            lobes.append((300.0, lerp(0.0, 0.62, (t - 18.0) / 3.0), 11))
        doa = dict(bearing_deg=round(bearing + 1.5 * math.sin(t * 5), 1),
                   confidence=round(0.66 + 0.2 * abs(math.sin(t)), 3),
                   valid=True, spectrum=srp_spectrum(lobes, t=t))
        person = dict(present=True,
                      bearing_deg=round(2.5 * math.sin(t * 1.7), 1),
                      distance_m=round(lerp(3.44, 1.15, k), 2),
                      score=round(0.86 + 0.06 * math.sin(t * 2), 2))
        state = dict(state="approach", mode=1,
                     cmd=dict(vx=round(lerp(0.38, 0.12, k), 2), vy=0.0,
                              wz=round(-0.04 * math.sin(t * 1.7), 2)),
                     target_bearing_deg=137.4,
                     distance_m=round(lerp(3.44, 1.15, k), 2))

    # ---- 24-27 s: arrived, standing in front of the person -----------------
    elif t < 27.0:
        doa = dict(bearing_deg=round(95.0 + 2.0 * math.sin(t * 3), 1),
                   confidence=0.7, valid=True,
                   spectrum=srp_spectrum([(95, 1.0, 12), (300, 0.5, 11)], t=t))
        person = dict(present=True, bearing_deg=round(1.2 * math.sin(t), 1),
                      distance_m=1.15, score=0.9)
        state = dict(state="stopped", mode=1, cmd=dict(vx=0.0, vy=0.0, wz=0.0),
                     target_bearing_deg=137.4, distance_m=1.15)

    # ---- 27-31 s: FAILURE 1 — person lost, bearings no longer trustworthy --
    elif t < 31.0:
        doa = dict(bearing_deg=round(95.0 + 60.0 * math.sin(t * 0.9), 1),
                   confidence=round(0.05 + 0.1 * abs(math.sin(t * 2)), 3),
                   valid=False, spectrum=srp_spectrum([], floor=0.5, t=t))
        person = dict(present=False)
        # An out-of-contract state, on purpose: the strip must show UNKNOWN.
        state = dict(state="recovering" if t > 29.0 else "stopped", mode=4,
                     cmd=dict(vx=0.0, vy=0.0, wz=0.0))

    # ---- 31-37 s: FAILURE 2 — total blackout. Nothing is published. --------
    elif t < 37.0:
        pass

    # ---- 37-42 s: recovery, and a doa message with NO spectrum key ---------
    else:
        doa = dict(bearing_deg=round(88.0 + 3.0 * math.sin(t * 3), 1),
                   confidence=0.64, valid=True)      # note: no "spectrum"
        person = dict(present=True, bearing_deg=-6.4, distance_m=1.4, score=0.79)
        state = dict(state="idle", mode=1, cmd=dict(vx=0.0, vy=0.0, wz=0.0))

    return doa, wake, person, state


# ---------------------------------------------------------------------------
# Node.
# ---------------------------------------------------------------------------
class MockDemo(Node):
    def __init__(self, args):
        super().__init__("mock_demo")
        self.args = args
        qos = 10
        self.pub = {
            "doa": self.create_publisher(String, args.prefix + "/doa", qos),
            "wake": self.create_publisher(String, args.prefix + "/wake", qos),
            "person": self.create_publisher(String, args.prefix + "/person", qos),
            "robot_state": self.create_publisher(
                String, args.prefix + "/robot_state", qos),
        }
        self.cam_pub = None
        if args.camera:
            self._init_camera(args.camera)

        self.t0 = time.time()
        self.tick = 0
        self.wake_fired = -1.0
        self.phase = ""
        # 10 Hz drives doa and person; robot_state is decimated to 5 Hz and the
        # camera to 4 Hz, so the dashboard sees three different arrival rates
        # and its staleness thresholds get tested against all of them.
        self.create_timer(0.1, self.on_tick)
        self.get_logger().info(
            f"mock demo publishing on {args.prefix}/* "
            f"({'looping' if not args.once else 'single pass'}, "
            f"speed x{args.speed:g})")

    # ---- optional synthetic camera ----------------------------------------
    def _init_camera(self, topic):
        """Publish a fake detector passthrough frame, so web.py's optional
        camera path can be exercised without atb. Imports are inside the
        function: this is a testing extra and must not stop the mock from
        running on a box with no Pillow."""
        try:
            from sensor_msgs.msg import CompressedImage
            importlib.import_module("PIL.ImageDraw")   # availability probe only
        except Exception as e:
            self.get_logger().error(f"--camera needs sensor_msgs + Pillow: {e}")
            return
        self._CompressedImage = CompressedImage
        self.cam_pub = self.create_publisher(CompressedImage, topic, 1)
        self.get_logger().info(f"mock camera on {topic}")

    def _publish_camera(self, person):
        import io as _io
        from PIL import Image, ImageDraw
        w, h = 640, 360
        img = Image.new("RGB", (w, h), (24, 30, 40))
        d = ImageDraw.Draw(img)
        for x in range(0, w, 64):                    # a floor grid, for parallax
            d.line([(x, h), (w // 2, h // 2)], fill=(38, 48, 62))
        d.line([(0, h // 2), (w, h // 2)], fill=(48, 60, 76))
        if person and person.get("present"):
            # Put the blob where the bearing says, using the same 69 deg FOV the
            # dashboard assumes, so the overlay bracket should land on it.
            b = person.get("bearing_deg", 0.0)
            cx = w * (0.5 - b / 69.0)
            dist = max(0.6, person.get("distance_m", 3.0))
            ph = min(h * 0.9, 260.0 / dist * 1.6)
            d.rectangle([cx - ph * 0.22, h * 0.55 - ph * 0.5,
                         cx + ph * 0.22, h * 0.55 + ph * 0.5], fill=(120, 150, 190))
            d.ellipse([cx - ph * 0.13, h * 0.55 - ph * 0.62,
                       cx + ph * 0.13, h * 0.55 - ph * 0.36], fill=(150, 175, 205))
        d.text((8, 8), f"mock passthrough {time.strftime('%H:%M:%S')}",
               fill=(200, 210, 220))
        bio = _io.BytesIO()
        img.save(bio, format="JPEG", quality=70)
        msg = self._CompressedImage()
        msg.format = "jpeg"
        msg.data = bio.getvalue()
        self.cam_pub.publish(msg)

    # ---- the loop ----------------------------------------------------------
    def send(self, topic, payload, stamp=None):
        d = dict(payload)
        d["v"] = SCHEMA_V
        d["topic"] = topic
        # Contract: stamp is taken at the ORIGIN of the data. The mock skews the
        # ROS 1 topics by a second on purpose (--skew) so that anything in the
        # dashboard that ages messages by `stamp` instead of by arrival shows up
        # immediately as a permanently-stale panel.
        d["stamp"] = round(stamp if stamp is not None else time.time(), 3)
        m = String()
        m.data = json.dumps(d)
        self.pub[topic].publish(m)

    def on_tick(self):
        self.tick += 1
        t = (time.time() - self.t0) * self.args.speed
        if self.args.once:
            if t > CYCLE_S:
                return                                   # go quiet and stay quiet
        else:
            t = t % CYCLE_S
            if t < 0.15:                                 # loop wrapped
                self.wake_fired = -1.0
        doa, _wake, person, state = scenario(t)

        # Phase logging, so a run in a terminal narrates what the dashboard
        # should be showing right now.
        phase = ("quiet" if t < 5 else "speech" if t < 6.5 else "turn" if t < 10
                 else "acquire" if t < 13 else "approach" if t < 24
                 else "stopped" if t < 27 else "person-lost/invalid" if t < 31
                 else "BLACKOUT" if t < 37 else "recovery(no spectrum)")
        if phase != self.phase:
            self.phase = phase
            self.get_logger().info(f"t={t:5.1f}s  {phase}")

        if doa is not None:
            self.send("doa", doa)
        if person is not None:
            self.send("person", person, stamp=time.time() - self.args.skew)
        if state is not None and self.tick % 2 == 0:      # 5 Hz per the contract
            self.send("robot_state", state, stamp=time.time() - self.args.skew)
        # The wake word fires once per pass, at the moment the speech phase ends.
        if 6.4 <= t < 6.6 and self.wake_fired < 0:
            self.wake_fired = t
            self.send("wake", dict(keyword="artaban", bearing_deg=137.4,
                                   confidence=0.82))
            self.get_logger().info("WAKE  artaban @ 137.4 deg")
        if self.cam_pub is not None and self.tick % 3 == 0 and t < 31:
            self._publish_camera(person)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--prefix", default="/dorai",
                    help="topic prefix (default /dorai)")
    ap.add_argument("--once", action="store_true",
                    help="run the scenario once, then stay silent")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="scenario time multiplier (default 1.0)")
    ap.add_argument("--skew", type=float, default=1.0,
                    help="seconds to backdate the ROS 1 stamps by, mimicking an "
                         "undisciplined clock on atb (default 1.0)")
    ap.add_argument("--camera", default="",
                    help="also publish a synthetic CompressedImage on this topic")
    # Split ROS args off ourselves: argparse must not choke on --ros-args.
    argv = list(sys.argv[1:] if argv is None else argv)
    ros_args = []
    if "--ros-args" in argv:
        i = argv.index("--ros-args")
        argv, ros_args = argv[:i], argv[i:]
    args = ap.parse_args(argv)

    rclpy.init(args=ros_args or None)
    node = MockDemo(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ExternalShutdownException is what rclpy raises when the process is
        # SIGTERMed (`timeout 30 ...`, `docker stop`, a CI harness). Catching it
        # keeps a routine "stop the mock" from printing a traceback that looks
        # like the mock crashed mid-scenario.
        pass
    except Exception:
        # The shutdown race differs between distros: on Humble a SIGTERM can
        # surface as a raw RCLError ("context is not valid") thrown from inside
        # the wait set rather than as ExternalShutdownException. If the context
        # is already gone we are on the way out anyway; anything else is a real
        # bug and still propagates.
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
