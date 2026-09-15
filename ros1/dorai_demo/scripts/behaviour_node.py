#!/usr/bin/env python3
"""
behaviour_node: the demo state machine.

    idle ---wake with confident bearing---> turn
    turn ---MEASURED yaw change reaches target---> acquire
    turn ---timeout with a large residual---> stopped
    acquire ---person seen---> approach          (timeout ---> idle)
    approach ---within standoff---> stopped      (person lost ---> stopped -> idle)
    stopped ---dwell---> idle

SAFETY. This node does not move the robot unless ~enable_motion is explicitly
true. With the default false it runs the whole state machine, publishes
/dorai/robot_state, and logs the twist it WOULD send, but never creates the
/robot_1/cmd_vel publisher at all. That is deliberate: the failure mode of a
half-tested behaviour node on a real quadruped in a room with people is not
recoverable, so arming it is a separate, conscious act by a human with a hand on
the stop.

Control facts used here come from DEMO_INTERFACES.md and are not rediscovered:
  * /robot_1/cmd_vel is geometry_msgs/Twist; only linear.x, linear.y and
    angular.z survive the local planner.
  * Publish at 20 Hz or faster; after 2.0 s of silence the planner zeroes the
    command. We run a fixed 25 Hz loop and publish every tick, zeros included,
    so the watchdog never fires mid-demo and the robot stands rather than
    freezing in an unknown state.
  * The planner scales what we send: forward 0.7, backward 0.35, strafe 0.3,
    yaw 1.0. Parameters here are in COMMANDED units, i.e. what we put on the
    wire, not what the robot ends up doing. A commanded 0.5 forward is about
    0.35 m/s on the ground.
  * A commanded magnitude below 0.05 makes the robot stand rather than step, so
    any intentional motion is floored above that; anything smaller is snapped to
    zero instead of being sent as a command that silently does nothing.
  * Mode 1 is READY. We gate all motion on having actually seen READY on
    /robot_1/control/state. We never publish to /robot_1/control/mode; the soft
    stop (publish 4) belongs to the human holding it.

CLOSED-LOOP YAW AND DISTANCE. The turn used to integrate the yaw rate it had
COMMANDED and assume the robot complied. It does not always: the planner
low-pass filters cmd_vel with a 0.1 per-message constant, so at our 25 Hz the
effective command lags the commanded one by roughly a second, and any foot slip
is invisible. Open loop the robot stops short while believing it finished, which
a small correction hides and a 90 degree turn does not.

So both dead-reckoned quantities now come from /robot_1/state/ground_truth
(quad_msgs/RobotState, 494 Hz, body.pose populated and tracking real heading):

  * the turn drives until the MEASURED yaw change reaches the target;
  * the blind leg of the approach measures how far the body has ACTUALLY moved.

Both keep the old commanded-value integral as a fallback, selected per tick by
whether the state topic is fresh, and both can be forced back to open loop with
~use_yaw_feedback / ~use_odom_distance if the estimator misbehaves mid-demo.

Yaw is unwrapped incrementally: each state message contributes wrap180 of its
change from the previous one to a continuous heading, and progress is that
continuous heading minus a value latched when the wake was accepted. At 494 Hz
the per-message change is a fraction of a degree, so the wrap is never
ambiguous, a turn through the +/-180 boundary accumulates normally, and a wrap
can never be mistaken for progress. Latching an absolute start yaw and taking
wrap180(now - start) instead would be ambiguous at exactly 180 degrees and would
silently fold anything past it.
"""
import collections
import importlib
import json
import math
import os
import subprocess

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Feedback comes from quad_msgs/RobotState. The type is known and verified, so
# bind it early rather than sniffing the connection header the way the control
# state does. If the package is missing (running outside a sourced xmas_ws) we
# say so once and run open loop, rather than dying: a node that will not start
# is worse than a node that runs the way it did last week.
try:
    from quad_msgs.msg import RobotState as _RobotState
except Exception as _exc:            # pragma: no cover - depends on the workspace
    _RobotState = None
    _ROBOT_STATE_IMPORT_ERROR = _exc
else:
    _ROBOT_STATE_IMPORT_ERROR = None

SCHEMA_VERSION = 1
MODE_READY = 1
MODE_UNKNOWN = -1

S_IDLE = "idle"
S_TURN = "turn"
S_ACQUIRE = "acquire"
S_APPROACH = "approach"
S_STOPPED = "stopped"


def wrap180(deg):
    """Fold an angle into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def yaw_from_quaternion(q):
    """Yaw in radians from a geometry_msgs/Quaternion.

    The body pitches and rolls while walking (a live sample read 0.042 in y),
    so use the full formula rather than the small-angle 2*atan2(z, w) shortcut,
    which is only exact when roll and pitch are zero.
    """
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def floor_magnitude(value, floor_):
    """Raise a non-zero command clear of the planner's dead band.

    Sending 0.02 is worse than sending 0.0: the robot stands, but the state
    machine believes it is moving. An exact zero stays zero; anything else is
    pushed out to `floor_` so it either moves the robot or is not sent at all.
    """
    if abs(value) < 1e-6:
        return 0.0
    if abs(value) < floor_:
        return math.copysign(floor_, value)
    return value


class BehaviourNode(object):
    def __init__(self):
        # -- safety ---------------------------------------------------------
        self.enable_motion = bool(rospy.get_param("~enable_motion", False))
        self.require_ready_mode = bool(rospy.get_param("~require_ready_mode", True))

        # -- rates ----------------------------------------------------------
        self.cmd_rate_hz = float(rospy.get_param("~cmd_rate_hz", 25.0))
        self.state_rate_hz = float(rospy.get_param("~state_rate_hz", 5.0))

        # -- wake gate ------------------------------------------------------
        # 0.15, down from 0.40 (2026-09-15). Every "artie" was recognised at
        # word confidence 1.00 and then rejected here: bearing confidences
        # measured 0.19, 0.28, 0.31, 0.35 and 0.36 on the robot. The number is
        # peak sharpness times the closure pass rate, and an isolated keyword
        # rarely fills the pooled window. The risk of a low value is bounded:
        # the bearing only chooses which way to turn, and the robot walks only
        # if the camera then finds a person; otherwise acquire times out.
        self.wake_min_confidence = float(rospy.get_param("~wake_min_confidence", 0.15))
        # 3.0 s: wake stamps are the end of the utterance, which Vosk endpoints
        # about 0.8 s late, plus bridge transit. 2.0 left too little margin.
        self.wake_max_age_s = float(rospy.get_param("~wake_max_age_s", 3.0))
        # Where the robot's forward axis sits in the ReSpeaker's 0..360 CCW
        # frame. Measure this once on the day; it is the single calibration
        # between the acoustic bearing and the robot's heading.
        # Where the microphone board's own 0 degrees points, expressed in the
        # board's frame, when the robot faces forward. The ReSpeaker is mounted
        # with its 0 pointing to the robot's LEFT, so the robot's heading sits
        # at 270 in board coordinates and a wake at board 270 needs no turn.
        # wrap180(bearing - 270) then yields a signed robot-relative angle,
        # positive to the left: board 0, the robot's left, maps to +90.
        self.mic_forward_deg = float(rospy.get_param("~mic_forward_deg", 270.0))

        # -- turn -----------------------------------------------------------
        # DEMO SPEED. 0.4 rad/s = 23 deg/s commanded, and the planner's yaw
        # scale is 1.0, so that is also what the legs are asked for. A 90 degree
        # turn takes about 4.5 s of saturated turning plus the tapered tail.
        # The last hardware test ran at 0.25 and was visibly sluggish.
        self.yaw_rate = float(rospy.get_param("~yaw_rate", 0.4))
        # Floor for the tapered tail of the turn. The planner steps only when
        # the norm of the SCALED command exceeds stand_cmd_vel_threshold 0.05;
        # yaw scale is 1.0, so 0.15 sits at 3x the threshold and unambiguously
        # steps. Raising it further would only add overshoot: after we stop
        # commanding, the planner's 0.1 filter constant at 25 Hz bleeds off over
        # roughly 0.4 s, carrying about 0.4*rate radians of extra turn, i.e.
        # ~3.4 deg at 0.15 - comfortably inside turn_tolerance_deg.
        self.yaw_min_rate = float(rospy.get_param("~yaw_min_rate", 0.15))
        self.yaw_kp = float(rospy.get_param("~yaw_kp", 1.2))
        self.turn_tolerance_deg = float(rospy.get_param("~turn_tolerance_deg", 8.0))
        self.turn_timeout_s = float(rospy.get_param("~turn_timeout_s", 12.0))
        # Drive the turn on the measured yaw rather than the commanded one.
        # Kept as a parameter so a human can fall back to the old open-loop
        # integral instantly, without an edit or a redeploy, if the state
        # estimator misbehaves mid-demo.
        self.use_yaw_feedback = bool(rospy.get_param("~use_yaw_feedback", True))
        # A turn that times out with more than this outstanding is a real
        # failure, not a rounding error, so it is logged at error level and ends
        # in `stopped` instead of being quietly handed to the camera.
        #
        # Why 20 degrees. The node itself already declares 25 the angle beyond
        # which it will not drive forwards at all (approach_max_bearing_deg): a
        # residual larger than that leaves the robot pointing somewhere the
        # approach controller considers "not towards the target", so acquire
        # cannot rescue it. 20 sits just inside that, and well outside the 8
        # degree success tolerance, so ordinary under- or overshoot never trips
        # it. It is also small in time terms: at yaw_rate 0.4 (23 deg/s) a
        # healthy turn covers 20 degrees in under a second, and the timeout is
        # 12 s, so 20 degrees still outstanding at timeout means the robot was
        # not turning, not that it was turning slowly.
        self.turn_fail_residual_deg = float(
            rospy.get_param("~turn_fail_residual_deg", 20.0))

        # -- acquire --------------------------------------------------------
        self.acquire_timeout_s = float(rospy.get_param("~acquire_timeout_s", 4.0))

        # -- approach -------------------------------------------------------
        self.standoff_m = float(rospy.get_param("~standoff_m", 0.7))
        self.approach_speed = float(rospy.get_param("~approach_speed", 0.5))
        self.approach_kp = float(rospy.get_param("~approach_kp", 0.4))
        # Floor for the tapered end of the approach, in COMMANDED units. The
        # planner steps only when the norm of the SCALED command clears 0.05,
        # and forward is scaled by 0.7, so the true commanded floor is
        # 0.05/0.7 = 0.071. The old 0.12 gave 0.084 scaled, only 1.7x the
        # threshold and reached slowly through the planner's 0.1 filter; the
        # operator's report that the robot responds badly to small commands is
        # exactly this. 0.20 gives 0.14 m/s on the ground, 2.8x the threshold.
        # Cost: the measured 0.78 s stop latency now carries ~0.11 m past the
        # standoff instead of ~0.065 m, so a 0.7 m standoff settles near 0.59 m.
        self.approach_min_speed = float(rospy.get_param("~approach_min_speed", 0.20))
        self.approach_yaw_kp = float(rospy.get_param("~approach_yaw_kp", 1.0))
        # Beyond this bearing error, turn on the spot instead of driving a long
        # curved path towards someone who is nearly beside the robot.
        self.approach_max_bearing_deg = float(
            rospy.get_param("~approach_max_bearing_deg", 25.0))
        # This does NOT need to tolerate dropped frames: person_node already
        # holds a target across a dropped frame for ~0.6 s and keeps republishing
        # present=true meanwhile. The two delays STACK, so the real blindness
        # after the camera last saw anybody is person_node's hold plus this.
        # Measured stop latency was 0.78 s with this at 0.8, giving 1.4 s total,
        # about 0.5 m of extra walking at the demo speed. At 0.4 the total is
        # ~1.0 s, and 0.4 s still tolerates four missed messages at 10 Hz.
        self.person_lost_timeout_s = float(
            rospy.get_param("~person_lost_timeout_s", 0.4))
        self.stopped_dwell_s = float(rospy.get_param("~stopped_dwell_s", 3.0))

        # -- dead reckoning after the person is lost -------------------------
        # The camera sits low, so a standing person leaves the TOP of the frame
        # as the robot closes in. Detection therefore works far and fails near,
        # the opposite of the usual failure. Measured: reliable around 3 m, gone
        # by roughly 1.8 m. The demo starts at 3 m and must finish at 0.7 m, so
        # losing the person on the way in is the NORMAL case, not an anomaly,
        # and stopping on loss would strand the robot a metre short every time.
        #
        # So on loss we keep going towards a remembered fix instead of stopping.
        # This is a deliberate trade: between the last sighting and the standoff
        # the robot is walking blind. Everything below exists to bound that.
        self.blind_approach = bool(rospy.get_param("~blind_approach", True))
        # Average the last samples rather than trusting the final one. The final
        # frame before a target is lost is the worst one to steer by: it is the
        # frame where the body was already half out of shot, so its centroid has
        # slid towards whatever part remained visible and its depth with it.
        self.memory_window_s = float(rospy.get_param("~memory_window_s", 1.0))
        self.memory_min_samples = int(rospy.get_param("~memory_min_samples", 3))
        # Hard ceilings on blind travel. The expected blind leg is about 1.1 m
        # (1.8 down to 0.7); anything much past that means the fix was wrong, so
        # stop rather than keep walking on a stale belief.
        self.max_blind_distance_m = float(
            rospy.get_param("~max_blind_distance_m", 2.0))
        self.max_blind_time_s = float(rospy.get_param("~max_blind_time_s", 12.0))
        # The local planner scales forward commands by 0.7 before they reach the
        # legs, so integrating the raw command would over-estimate progress and
        # stop the robot short. Dead reckoning has to use the scaled value.
        # This is only the FALLBACK path now; see ~use_odom_distance.
        self.planner_forward_scale = float(
            rospy.get_param("~planner_forward_scale", 0.7))
        # Measure the blind leg from the body position in the state message
        # instead of integrating the commanded speed. Same reasoning, and same
        # instant fallback, as ~use_yaw_feedback.
        self.use_odom_distance = bool(rospy.get_param("~use_odom_distance", True))

        # -- body state feedback ---------------------------------------------
        self.body_state_topic = str(rospy.get_param(
            "~body_state_topic", "/robot_1/state/ground_truth"))
        # How long without a state message before we stop believing it. The
        # topic runs at 494 Hz, so 0.5 s is 250 missed messages: unambiguously
        # dead rather than scheduling jitter. It is also short enough to bound
        # the damage, because a frozen measurement means the turn keeps
        # commanding while reading no progress: at yaw_rate 0.4 that is at most
        # 0.5*0.4 = 0.2 rad = 11.5 deg of unmeasured turn before the open-loop
        # fallback takes over, which stays inside turn_fail_residual_deg.
        self.state_stale_s = float(rospy.get_param("~state_stale_s", 0.5))

        # The planner's dead band is 0.05 on the norm of the SCALED command, not
        # on what we put on the wire. Forward is the worst case at scale 0.7, so
        # a commanded 0.05 arrives as 0.035 and the robot stands: the old
        # default sat BELOW the very band it was meant to clear. 0.05/0.7 =
        # 0.071 is the real forward floor; 0.10 clears it by 40% and clears the
        # yaw band (scale 1.0) twice over.
        self.min_cmd_magnitude = float(rospy.get_param("~min_cmd_magnitude", 0.10))

        # -- arrival announcement ---------------------------------------------
        # Spoken ONLY when the approach genuinely reached the standoff, by sight
        # or by dead reckoning. Every other way into `stopped` (a failed turn, a
        # person lost with no usable fix, a blind leg that timed out) stays
        # silent: a robot announcing it is ready to assist straight after
        # failing to find anybody is worse than one that says nothing.
        #
        # The WAV is pre-rendered (espeak-ng) and ships in this package, so the
        # robot needs no text-to-speech engine. Playback goes through aplay,
        # the same path the existing "bark" service uses.
        #
        # The device is pinned by the card's NAME. The ALSA default does not
        # reach the speakers on atb (see web_mod's play_device note), and card
        # NUMBERS move across reboots, so never hw:0,0.
        #
        # The ReSpeaker hears the phrase. That is accepted: it does not contain
        # the wake keyword, and the gate needs the keyword alone, so it cannot
        # retrigger the robot; the dashboard dial will briefly point at the
        # robot's own speaker.
        self.speak_on_arrival = bool(rospy.get_param("~speak_on_arrival", True))
        self.arrival_sound = str(rospy.get_param(
            "~arrival_sound", self._default_sound("ready_to_assist.wav")))
        self.audio_device = str(rospy.get_param(
            "~audio_device", "plughw:CARD=PCH,DEV=0"))
        self._player = None

        # -- state ----------------------------------------------------------
        self.state = S_IDLE
        self.state_since = rospy.Time.now().to_sec()
        self.mode = MODE_UNKNOWN
        self.mode_stamp = 0.0
        self.target_bearing_deg = None   # acoustic bearing that triggered us
        self.turn_remaining_deg = 0.0
        self.last_tick = None
        self.cmd = Twist()

        self.person = None               # last person payload with present true
        self.person_wall = 0.0
        self.person_hist = collections.deque(maxlen=64)  # (wall, dist, bearing)
        self.blind = None                # dead-reckoned fix, see enter_blind()

        # -- measured body state ---------------------------------------------
        # yaw_unwrapped_deg is a continuous heading: every message adds wrap180
        # of its change from the previous one, so it never jumps at +/-180 and
        # a full revolution reads as 360, not 0.
        self.yaw_unwrapped_deg = 0.0
        self.yaw_last_raw_deg = None
        self.body_xy = None
        self.state_msg_wall = 0.0
        # Bumped whenever the stream breaks for longer than state_stale_s.
        # Anything holding a reference into the stream compares epochs and
        # re-anchors rather than crediting the gap twice: the open-loop
        # fallback already covered it.
        self.state_epoch = 0
        self._yaw_ref = {}               # key -> (epoch, unwrapped yaw deg)
        # Turn bookkeeping, for a completion log that says which source the
        # robot actually turned on.
        self.turn_target_deg = 0.0
        self.turn_measured_deg = 0.0
        self.turn_openloop_deg = 0.0
        # True once this turn has actually been permitted to send a command. A
        # residual only convicts the robot if we were allowed to ask it to move;
        # in a dry run, or while the control mode is not READY, the robot is
        # standing still because we told it nothing, which is not a failure.
        self.turn_commanded = False

        if _RobotState is None and (self.use_yaw_feedback
                                    or self.use_odom_distance):
            rospy.logerr("behaviour_node: cannot import quad_msgs/RobotState "
                         "(%s); yaw and distance feedback DISABLED, falling "
                         "back to commanded-value dead reckoning",
                         _ROBOT_STATE_IMPORT_ERROR)
            self.use_yaw_feedback = False
            self.use_odom_distance = False

        self._control_state_cls = None

        # -- io -------------------------------------------------------------
        self.pub_state = rospy.Publisher("/dorai/robot_state", String, queue_size=5)
        self.pub_cmd = None
        if self.enable_motion:
            self.pub_cmd = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=1)
            rospy.logwarn("behaviour_node: MOTION ENABLED, /robot_1/cmd_vel is live")
        else:
            rospy.logwarn("behaviour_node: motion DISABLED (~enable_motion=false); "
                          "computing and reporting only, cmd_vel not advertised")

        rospy.Subscriber("/dorai/wake", String, self.on_wake, queue_size=5)
        rospy.Subscriber("/dorai/person", String, self.on_person, queue_size=5)
        if _RobotState is not None:
            # queue_size 1: at 494 Hz we only ever want the newest pose, and a
            # backlog would make the "is it fresh" test meaningless.
            rospy.Subscriber(self.body_state_topic, _RobotState,
                             self.on_body_state, queue_size=1)
            rospy.loginfo("behaviour_node: yaw feedback %s, odometry distance "
                          "%s, from %s",
                          "ON" if self.use_yaw_feedback else "off",
                          "ON" if self.use_odom_distance else "off",
                          self.body_state_topic)
        self.subscribe_control_state(
            rospy.get_param("~control_state_topic", "/robot_1/control/state"))

        rospy.Timer(rospy.Duration(1.0 / self.cmd_rate_hz), self.on_tick)
        rospy.Timer(rospy.Duration(1.0 / self.state_rate_hz), self.on_publish_state)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("behaviour_node up in state %s", self.state)

    # ---- control mode ------------------------------------------------------

    def subscribe_control_state(self, topic):
        """Subscribe without hard-coding the message type.

        quad_sdk's control state type is not pinned in the contract and the
        walking stack was not running to check, so bind late: take the type off
        the connection header of the first message and deserialize from then on.
        """
        rospy.Subscriber(topic, rospy.AnyMsg, self.on_control_state, queue_size=1)
        rospy.loginfo("behaviour_node gating motion on %s", topic)

    def on_control_state(self, msg):
        if self._control_state_cls is None:
            try:
                type_str = msg._connection_header["type"]
                pkg, cls_name = type_str.split("/")
                module = importlib.import_module(pkg + ".msg")
                self._control_state_cls = getattr(module, cls_name)
                rospy.loginfo("behaviour_node: control state type is %s", type_str)
            except Exception as exc:
                rospy.logwarn_throttle(10.0,
                                       "cannot resolve control state type: %s", exc)
                return
        try:
            parsed = self._control_state_cls().deserialize(msg._buff)
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "cannot parse control state: %s", exc)
            return

        value = getattr(parsed, "data", None)
        if value is None:
            value = getattr(parsed, "mode", None)
        if value is None:
            return
        self.mode = int(value)
        self.mode_stamp = rospy.Time.now().to_sec()

    def motion_allowed(self):
        if not self.enable_motion:
            return False
        if not self.require_ready_mode:
            return True
        if self.mode != MODE_READY:
            rospy.logwarn_throttle(
                5.0, "behaviour_node holding still: control mode is %s, need %d",
                self.mode, MODE_READY)
            return False
        return True

    # ---- measured body state -----------------------------------------------

    def on_body_state(self, msg):
        """Fold one 494 Hz state message into a continuous heading and position.

        Deliberately does almost nothing: the control decisions all happen on
        the 25 Hz tick, and doing them here would run them 20 times per tick.
        """
        now = rospy.Time.now().to_sec()
        if self.state_msg_wall and (now - self.state_msg_wall) > self.state_stale_s:
            # The stream broke and came back. Everything holding a reference
            # into it must re-anchor rather than book the whole gap as progress.
            self.state_epoch += 1
        pose = msg.body.pose
        yaw = math.degrees(yaw_from_quaternion(pose.orientation))
        if self.yaw_last_raw_deg is None:
            self.yaw_unwrapped_deg = yaw
        else:
            # wrap180 of the per-message change. At 494 Hz and a demo yaw rate
            # of 0.4 rad/s this is about 0.05 deg, so the fold is never
            # ambiguous and crossing +/-180 costs nothing.
            self.yaw_unwrapped_deg += wrap180(yaw - self.yaw_last_raw_deg)
        self.yaw_last_raw_deg = yaw
        self.body_xy = (pose.position.x, pose.position.y)
        self.state_msg_wall = now

    def state_fresh(self):
        """True when the body state is recent enough to steer on."""
        if self.state_msg_wall <= 0.0:
            return False
        return (rospy.Time.now().to_sec() - self.state_msg_wall) <= self.state_stale_s

    def consume_yaw_delta(self, key):
        """Measured yaw change, in degrees, since the last call under `key`.

        Returns None when the measurement must not be used, which is the signal
        for the caller to fall back to integrating what it commanded. That
        happens when feedback is switched off and when the state topic has gone
        stale.

        WHAT HAPPENS WHEN THE FEEDBACK GOES AWAY, and why. A stale topic falls
        back to the old open-loop integral and warns; it does not stop the
        robot and it does not keep turning on a frozen number. Stopping was
        rejected because a dropped estimator mid-turn would abort the demo in
        front of an audience for a fault the robot can ride out: open loop is
        exactly how this worked last week, on hardware, end to end. Continuing
        on the frozen value was rejected because the residual would never fall,
        so the turn would run to its full 12 s timeout at full rate - the one
        genuinely dangerous outcome. The fallback is still bounded by
        turn_timeout_s, and the first sample after the topic returns is used
        only to re-anchor, never credited, because the open-loop integral has
        already accounted for that window.
        """
        if not self.use_yaw_feedback:
            return None
        if not self.state_fresh():
            rospy.logwarn_throttle(
                2.0, "body state stale (%.1f s since last message on %s): "
                "falling back to open-loop dead reckoning",
                rospy.Time.now().to_sec() - self.state_msg_wall
                if self.state_msg_wall else float("inf"),
                self.body_state_topic)
            self._yaw_ref.pop(key, None)
            return None
        ref = self._yaw_ref.get(key)
        self._yaw_ref[key] = (self.state_epoch, self.yaw_unwrapped_deg)
        if ref is None or ref[0] != self.state_epoch:
            return 0.0
        return self.yaw_unwrapped_deg - ref[1]

    def latch_yaw_ref(self, key):
        """Anchor `key` to the heading right now, or clear it if there is none."""
        if self.use_yaw_feedback and self.state_fresh():
            self._yaw_ref[key] = (self.state_epoch, self.yaw_unwrapped_deg)
            return True
        self._yaw_ref.pop(key, None)
        return False

    def odom_travel(self, blind):
        """Update blind["travelled_m"] from the measured body position.

        Returns True when the measurement was used, False when the caller must
        integrate the commanded speed instead.

        Distance is the straight-line displacement from an anchor rather than a
        sum of per-sample steps. Summing steps at 25 Hz turns position noise
        into a one-way random walk that only ever adds progress, which is the
        wrong direction to be wrong in: it would stop the robot short of the
        person. Displacement from a fixed anchor has no such bias. It does
        under-read a curved path, but the blind leg is the last metre of a
        straight walk-in, so the difference is small and, again, conservative.
        """
        if not self.use_odom_distance or not self.state_fresh() or self.body_xy is None:
            blind["anchor_xy"] = None      # re-anchor when the stream comes back
            return False
        if blind["anchor_xy"] is None or blind["anchor_epoch"] != self.state_epoch:
            # First fix of this run: keep whatever was dead-reckoned so far and
            # measure from here on, so a gap is never counted twice.
            blind["anchor_xy"] = self.body_xy
            blind["anchor_epoch"] = self.state_epoch
            blind["anchor_travelled_m"] = blind["travelled_m"]
            return True
        blind["travelled_m"] = blind["anchor_travelled_m"] + math.hypot(
            self.body_xy[0] - blind["anchor_xy"][0],
            self.body_xy[1] - blind["anchor_xy"][1])
        return True

    # ---- inputs ------------------------------------------------------------

    def on_wake(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        if self.state != S_IDLE:
            rospy.loginfo("wake ignored, busy in %s", self.state)
            return

        confidence = float(payload.get("confidence", 0.0))
        bearing = payload.get("bearing_deg")
        stamp = float(payload.get("stamp", 0.0))
        age = rospy.Time.now().to_sec() - stamp

        if bearing is None:
            rospy.logwarn("wake without a bearing, ignoring")
            return
        if confidence < self.wake_min_confidence:
            rospy.loginfo("wake confidence %.2f below %.2f, ignoring",
                          confidence, self.wake_min_confidence)
            return
        if stamp > 0.0 and age > self.wake_max_age_s:
            rospy.logwarn("wake is %.1f s stale, ignoring", age)
            return

        relative = wrap180(float(bearing) - self.mic_forward_deg)
        self.target_bearing_deg = float(bearing)
        self.turn_remaining_deg = relative
        self.turn_target_deg = relative
        self.turn_measured_deg = 0.0
        self.turn_openloop_deg = 0.0
        self.turn_commanded = False
        # Latch the heading at the moment the wake is ACCEPTED, not at the first
        # tick of the turn: up to 40 ms of tick period sits between the two, and
        # anything the robot did in that window would otherwise be free.
        latched = self.latch_yaw_ref("turn")
        rospy.loginfo("WAKE %r conf %.2f at %.1f deg -> turn %.1f deg (%s)",
                      payload.get("keyword"), confidence, bearing, relative,
                      "closed loop on measured yaw" if latched
                      else "OPEN LOOP, no usable body state")
        self.enter(S_TURN)

    def on_person(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        if payload.get("present"):
            self.person = payload
            self.person_wall = rospy.Time.now().to_sec()
            try:
                self.person_hist.append((self.person_wall,
                                         float(payload["distance_m"]),
                                         float(payload["bearing_deg"])))
            except (KeyError, TypeError, ValueError):
                # present=true without usable numbers: keep the freshness stamp
                # so the target does not flicker, but never feed junk into a
                # fix the robot will later walk towards while blind.
                pass

    def averaged_fix(self):
        """Mean distance and bearing over the last `memory_window_s` of sightings.

        Returns (distance_m, bearing_deg, age_s) or None when there is not
        enough history to be worth trusting. A plain mean is right here: the
        bearings are signed, confined to the camera's own field of view, and
        never wrap, so none of the circular-mean machinery applies.

        The age matters as much as the values. The fix describes where the
        person was when it was taken, not where they are now, and the caller has
        to subtract the distance covered since then or it will stop short.
        """
        if not self.person_hist:
            return None
        newest = self.person_hist[-1][0]
        rows = [r for r in self.person_hist if newest - r[0] <= self.memory_window_s]
        if len(rows) < self.memory_min_samples:
            return None
        n = float(len(rows))
        return (sum(r[1] for r in rows) / n,
                sum(r[2] for r in rows) / n,
                rospy.Time.now().to_sec() - newest)

    def enter_blind(self):
        """Latch an averaged fix and switch the approach to dead reckoning."""
        fix = self.averaged_fix()
        if fix is None:
            rospy.logwarn("person lost with too little history to remember, stopping")
            self.enter(S_STOPPED)
            return False
        distance, bearing, age = fix
        # The fix is already `age` seconds old and the robot has been walking
        # the whole time, so bring it up to date before using it as a budget.
        travelled = self.approach_speed * self.planner_forward_scale * age
        remaining = distance - travelled - self.standoff_m
        if remaining <= 0.0:
            rospy.loginfo("person lost but already inside standoff, stopping")
            self.enter(S_STOPPED)
            return False
        # max_blind_distance_m caps the budget, so measuring travel against the
        # budget below is also what enforces the hard ceiling on blind travel.
        remaining = min(remaining, self.max_blind_distance_m)
        self.blind = {"budget_m": remaining, "travelled_m": 0.0,
                      "bearing_deg": bearing,
                      "since": rospy.Time.now().to_sec(),
                      "anchor_xy": None, "anchor_epoch": -1,
                      "anchor_travelled_m": 0.0}
        self.latch_yaw_ref("blind")
        measured = self.odom_travel(self.blind)
        rospy.logwarn("person lost: dead reckoning %.2f m at %.1f deg "
                      "from a %d-sample fix (%.2f m, %.1f deg, %.2f s old), "
                      "distance measured by %s",
                      remaining, bearing, len(self.person_hist), distance,
                      bearing, age,
                      "odometry" if measured else "COMMANDED SPEED")
        return True

    def person_fresh(self):
        return (self.person is not None
                and (rospy.Time.now().to_sec() - self.person_wall)
                <= self.person_lost_timeout_s)

    # ---- state machine -----------------------------------------------------

    @staticmethod
    def _default_sound(name):
        """Path of a sound shipped in this package's sounds/ directory."""
        try:
            import rospkg
            return os.path.join(rospkg.RosPack().get_path("dorai_demo"),
                                "sounds", name)
        except Exception:
            # Fall back to the script's own location. realpath follows the
            # devel/ symlink back to the source tree, where sounds/ lives.
            here = os.path.dirname(os.path.realpath(__file__))
            return os.path.join(here, os.pardir, "sounds", name)

    def announce_arrival(self):
        """Play the arrival phrase without blocking the 25 Hz control loop.

        Fire and forget: aplay runs as a child process and the tick carries on.
        A failure to play is logged and never allowed to affect control; the
        robot has already stopped by the time this matters.
        """
        if not self.speak_on_arrival:
            return
        if self._player is not None and self._player.poll() is None:
            return  # still speaking; do not stack a second copy on top
        if not os.path.isfile(self.arrival_sound):
            rospy.logwarn("arrival sound not found at %s, staying silent",
                          self.arrival_sound)
            return
        try:
            self._player = subprocess.Popen(
                ["aplay", "-q", "-D", self.audio_device, self.arrival_sound],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rospy.loginfo("announcing arrival on %s", self.audio_device)
        except OSError as exc:
            rospy.logwarn("could not start aplay (%s), staying silent", exc)

    def enter(self, state):
        if state == self.state:
            return
        rospy.loginfo("state %s -> %s", self.state, state)
        self.state = state
        self.state_since = rospy.Time.now().to_sec()

    def elapsed(self):
        return rospy.Time.now().to_sec() - self.state_since

    def on_tick(self, _evt):
        now = rospy.Time.now().to_sec()
        dt = 0.0 if self.last_tick is None else (now - self.last_tick)
        self.last_tick = now
        dt = clamp(dt, 0.0, 0.2)   # a scheduling hiccup must not fake progress

        cmd = Twist()

        if self.state == S_IDLE:
            pass

        elif self.state == S_TURN:
            # Retire whatever the robot has actually done since the last tick
            # BEFORE deciding whether the turn is finished, so the completion
            # test always reads the freshest measurement.
            measured = self.consume_yaw_delta("turn")
            if measured is not None:
                self.turn_remaining_deg -= measured
                self.turn_measured_deg += measured

            remaining = self.turn_remaining_deg
            if abs(remaining) <= self.turn_tolerance_deg:
                rospy.loginfo(
                    "turn complete, residual %.1f deg of %.1f (measured %.1f, "
                    "open loop %.1f)", remaining, self.turn_target_deg,
                    self.turn_measured_deg, self.turn_openloop_deg)
                if abs(self.turn_openloop_deg) > abs(self.turn_measured_deg):
                    rospy.logwarn("that turn finished mostly on the OPEN-LOOP "
                                  "estimate; the robot may not have got there")
                self.enter(S_ACQUIRE)
            elif self.elapsed() > self.turn_timeout_s:
                # A timeout used to fall through to acquire with the whole angle
                # still outstanding, silently making it the camera's problem.
                # With a real measurement behind it, a large residual here means
                # the robot did not turn, and walking on would send it at
                # whoever happens to be in front of it instead of whoever spoke.
                if (abs(remaining) > self.turn_fail_residual_deg
                        and self.turn_commanded):
                    rospy.logerr(
                        "TURN FAILED: %.1f s timeout with %.1f deg of %.1f "
                        "still outstanding (measured %.1f deg of actual "
                        "movement). Not continuing to acquire; stopping.",
                        self.turn_timeout_s, remaining, self.turn_target_deg,
                        self.turn_measured_deg)
                    self.enter(S_STOPPED)
                elif not self.turn_commanded:
                    rospy.logwarn(
                        "turn timed out with %.1f deg remaining, but nothing "
                        "was ever sent to the robot (motion disabled or mode "
                        "not READY), so this is not a turn failure",
                        remaining)
                    self.enter(S_ACQUIRE)
                else:
                    rospy.logwarn(
                        "turn timed out with %.1f deg remaining, inside the "
                        "%.1f deg failure threshold: handing the rest to the "
                        "camera", remaining, self.turn_fail_residual_deg)
                    self.enter(S_ACQUIRE)
            else:
                rate = clamp(self.yaw_kp * math.radians(abs(remaining)),
                             self.yaw_min_rate, self.yaw_rate)
                rate = math.copysign(rate, remaining)
                cmd.angular.z = rate
                if self.motion_allowed():
                    self.turn_commanded = True
                if measured is None and self.motion_allowed():
                    # Fallback only: no usable measurement this tick, so assume
                    # the robot complied, exactly as this node did before the
                    # loop was closed. Integrate only what we are allowed to
                    # send, and keep the total separately so the completion log
                    # can say how much of the turn was believed rather than
                    # seen.
                    step = math.degrees(rate * dt)
                    self.turn_remaining_deg -= step
                    self.turn_openloop_deg += step

        elif self.state == S_ACQUIRE:
            if self.person_fresh():
                rospy.loginfo("person acquired at %.2f m, %.1f deg",
                              self.person["distance_m"], self.person["bearing_deg"])
                self.blind = None
                self.enter(S_APPROACH)
            elif self.elapsed() > self.acquire_timeout_s:
                # Give up rather than walk blind towards a sound.
                rospy.logwarn("no person within %.1f s, giving up",
                              self.acquire_timeout_s)
                self.enter(S_IDLE)

        elif self.state == S_APPROACH:
            if not self.person_fresh() and self.blind is None:
                if self.blind_approach:
                    self.enter_blind()
                else:
                    rospy.logwarn("person lost during approach, stopping")
                    self.enter(S_STOPPED)

            if self.state == S_APPROACH and self.blind is not None:
                # Take the measurements first, so the budget test below sees
                # where the robot actually is rather than where it was told to
                # be. Both fall back to the commanded value further down when
                # the state topic is unusable.
                odom_ok = self.odom_travel(self.blind)
                measured_yaw = self.consume_yaw_delta("blind")
                if measured_yaw is not None:
                    # The remembered fix is a bearing in the robot's own frame,
                    # so every degree the body turns moves it. Measured yaw is
                    # a strictly better source for that than commanded yaw, for
                    # the same reason it is in the turn.
                    self.blind["bearing_deg"] -= measured_yaw
                remaining_m = self.blind["budget_m"] - self.blind["travelled_m"]

                # Seeing the person again always beats a remembered fix.
                if self.person_fresh():
                    rospy.loginfo("person reacquired, leaving dead reckoning "
                                  "with %.2f m still budgeted", remaining_m)
                    self.blind = None
                elif remaining_m <= 0.0:
                    rospy.loginfo("dead-reckoned standoff reached: %.2f m "
                                  "travelled of a %.2f m budget, by %s",
                                  self.blind["travelled_m"],
                                  self.blind["budget_m"],
                                  "odometry" if odom_ok else "commanded speed")
                    self.enter(S_STOPPED)
                    self.announce_arrival()
                elif (now - self.blind["since"]) > self.max_blind_time_s:
                    rospy.logwarn("blind approach timed out with %.2f m left",
                                  remaining_m)
                    self.enter(S_STOPPED)
                else:
                    bearing = self.blind["bearing_deg"]
                    cmd.angular.z = floor_magnitude(
                        clamp(self.approach_yaw_kp * math.radians(bearing),
                              -self.yaw_rate, self.yaw_rate),
                        self.min_cmd_magnitude)
                    if abs(bearing) > self.approach_max_bearing_deg:
                        cmd.linear.x = 0.0
                    else:
                        cmd.linear.x = floor_magnitude(
                            self.approach_speed, self.min_cmd_magnitude)
                    if self.motion_allowed():
                        # Fallbacks, each used only when its measurement is
                        # unavailable this tick. Integrate only what is actually
                        # sent, and for distance only the part that survives the
                        # planner's 0.7 forward scale.
                        if not odom_ok:
                            self.blind["travelled_m"] += (
                                cmd.linear.x * self.planner_forward_scale * dt)
                        if measured_yaw is None:
                            self.blind["bearing_deg"] -= math.degrees(
                                cmd.angular.z * dt)

            if self.state == S_APPROACH and self.blind is None:
                distance = float(self.person["distance_m"])
                bearing = float(self.person["bearing_deg"])
                if distance <= self.standoff_m:
                    rospy.loginfo("standoff reached at %.2f m", distance)
                    self.enter(S_STOPPED)
                    self.announce_arrival()
                else:
                    cmd.angular.z = floor_magnitude(
                        clamp(self.approach_yaw_kp * math.radians(bearing),
                              -self.yaw_rate, self.yaw_rate),
                        self.min_cmd_magnitude)
                    if abs(bearing) > self.approach_max_bearing_deg:
                        cmd.linear.x = 0.0   # turn on the spot first
                    else:
                        speed = clamp(self.approach_kp * (distance - self.standoff_m),
                                      self.approach_min_speed, self.approach_speed)
                        cmd.linear.x = floor_magnitude(speed, self.min_cmd_magnitude)

        elif self.state == S_STOPPED:
            if self.elapsed() > self.stopped_dwell_s:
                self.person = None
                self.target_bearing_deg = None
                # A remembered fix belongs to one approach only. Carrying it
                # into the next wake would send the robot walking towards where
                # somebody stood minutes ago.
                self.blind = None
                self.person_hist.clear()
                self.enter(S_IDLE)

        self.cmd = cmd
        self.send(cmd)

    def send(self, cmd):
        if not self.motion_allowed():
            if abs(cmd.linear.x) > 0.0 or abs(cmd.angular.z) > 0.0:
                rospy.loginfo_throttle(
                    1.0, "[dry run] would send vx=%.3f wz=%.3f in %s",
                    cmd.linear.x, cmd.angular.z, self.state)
            return
        # Every tick, zeros included: the planner zeroes after 2.0 s of silence
        # and we would rather the robot stand than time out.
        self.pub_cmd.publish(cmd)

    # ---- output ------------------------------------------------------------

    def on_publish_state(self, _evt):
        payload = {
            "v": SCHEMA_VERSION,
            "topic": "robot_state",
            "stamp": round(rospy.Time.now().to_sec(), 3),
            "state": self.state,
            "mode": self.mode,
            "cmd": {
                "vx": round(self.cmd.linear.x, 3),
                "vy": round(self.cmd.linear.y, 3),
                "wz": round(self.cmd.angular.z, 3),
            },
        }
        if self.target_bearing_deg is not None:
            payload["target_bearing_deg"] = round(self.target_bearing_deg, 2)
        if self.person_fresh():
            payload["distance_m"] = round(float(self.person["distance_m"]), 3)
        self.pub_state.publish(String(data=json.dumps(payload)))

    def shutdown(self):
        """Leave the robot standing, not coasting on a stale command."""
        if self.pub_cmd is not None:
            stop = Twist()
            for _ in range(5):
                self.pub_cmd.publish(stop)
                rospy.sleep(0.02)
            rospy.loginfo("behaviour_node sent zero velocity on shutdown")


if __name__ == "__main__":
    rospy.init_node("behaviour_node")
    BehaviourNode()
    rospy.spin()
