#!/usr/bin/env python3
"""
person_node: Detection3DArray -> the single /dorai/person message in the contract.

Scope, agreed and deliberately small: keep class 0, drop low confidence and
implausible distances, pick the nearest survivor, hold briefly across a dropped
frame. No tracking, no identity across frames.

Geometry. The camera publishes spatial coordinates in the colour optical frame,
already in metres: z forward, x right, y down. The contract's person bearing is
signed and positive to the LEFT of the robot's forward axis, so

    bearing_deg = atan2(-x, z)

No intrinsics are involved. Both quantities are already metric, which is the
whole point of using the spatial detection output rather than pixel boxes.

Note this bearing is NOT the same quantity as the acoustic bearing on
/dorai/doa. That one is measured from the microphone board's reference; this one
from where the robot is pointing. The contract says do not conflate them.
"""
import json
import math

import rospy
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

SCHEMA_VERSION = 1

# Which class index means "person" depends on the model, and the two candidate
# models here disagree:
#   YOLOv4-tiny COCO   person = 0   (80 classes)
#   MobileNet-SSD VOC  person = 15  (21 classes, index 0 is "background")
# Hard-coding 0 silently turns a MobileNet run into "never sees anybody", so it
# is a parameter. See detection_override.launch for which model is in use.
DEFAULT_PERSON_CLASS_ID = 0


class PersonNode(object):
    def __init__(self):
        self.person_class_id = int(
            rospy.get_param("~person_class_id", DEFAULT_PERSON_CLASS_ID))
        # Measured on the robot: with a 0.5 floor, this camera reported a
        # person on an EMPTY workbench 43 percent of the time, scoring 0.50 to
        # 0.73 (mean 0.57). Real people scored 0.76 to 0.99 in the same blob.
        # 0.6 is the compromise; the distance floor below does the heavy work.
        self.min_score = rospy.get_param("~min_score", 0.6)
        # The stereo threshold filter is configured 0.3 m to 5.0 m, so nothing
        # beyond 5 m exists at all. A detection reporting more than that is a
        # depth failure, not a distant person.
        # The false detections above all sat at a fixed 0.52 m, which is
        # clutter close to the lens rather than anybody real. Nobody the robot
        # is meant to walk towards is closer than a metre, and the standoff is
        # 1.5 m anyway, so a 1.0 m floor removes that whole class of nonsense
        # without discarding a single plausible target.
        self.min_distance_m = rospy.get_param("~min_distance_m", 1.0)
        self.max_distance_m = rospy.get_param("~max_distance_m", 5.0)
        # Reject absurd lateral offsets, which show up when the bbox ROI lands
        # on a depth hole and the driver returns a garbage centroid.
        self.max_abs_bearing_deg = rospy.get_param("~max_abs_bearing_deg", 80.0)
        # One dropped frame at ~10 Hz must not make the target vanish and stall
        # the behaviour node mid-approach.
        self.hold_s = rospy.get_param("~hold_s", 0.6)
        self.publish_rate_hz = rospy.get_param("~publish_rate_hz", 10.0)
        self.detections_topic = rospy.get_param(
            "~detections_topic", "/atb_front/nn/spatial_detections")

        self._last = None          # dict payload of the last accepted person
        self._last_wall = 0.0      # wall clock of that acceptance, for the hold
        self._was_present = False

        self.pub = rospy.Publisher("/dorai/person", String, queue_size=5)
        rospy.Subscriber(self.detections_topic, Detection3DArray,
                         self.on_detections, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self.on_tick)

        rospy.loginfo("person_node up: %s -> /dorai/person at %.1f Hz "
                      "(person_class_id=%d min_score=%.2f band=%.1f..%.1f m "
                      "hold=%.2f s)",
                      self.detections_topic, self.publish_rate_hz,
                      self.person_class_id, self.min_score, self.min_distance_m,
                      self.max_distance_m, self.hold_s)

    def on_detections(self, msg):
        best = None
        for det in msg.detections:
            if not det.results:
                continue
            res = det.results[0]
            if int(res.id) != self.person_class_id:
                continue
            if float(res.score) < self.min_score:
                continue

            p = res.pose.pose.position
            # Horizontal range. y is height and irrelevant to how far away the
            # person is on the floor, so it is deliberately excluded.
            distance = math.hypot(p.x, p.z)
            if not (self.min_distance_m <= distance <= self.max_distance_m):
                continue
            if p.z <= 0.0:
                continue  # behind the image plane, impossible

            bearing = math.degrees(math.atan2(-p.x, p.z))
            if abs(bearing) > self.max_abs_bearing_deg:
                continue

            if best is None or distance < best[0]:
                best = (distance, bearing, float(res.score))

        if best is None:
            return

        distance, bearing, score = best
        # Origin stamp, per the contract: when the data came into being, not
        # when we got round to sending it.
        stamp = msg.header.stamp.to_sec()
        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()

        self._last = {
            "v": SCHEMA_VERSION,
            "topic": "person",
            "stamp": round(stamp, 3),
            "present": True,
            "bearing_deg": round(bearing, 2),
            "distance_m": round(distance, 3),
            "score": round(score, 3),
        }
        self._last_wall = rospy.Time.now().to_sec()

    def on_tick(self, _evt):
        now = rospy.Time.now().to_sec()
        fresh = (self._last is not None
                 and (now - self._last_wall) <= self.hold_s)

        if fresh:
            payload = dict(self._last)
        else:
            # When present is false the contract omits the other fields.
            payload = {
                "v": SCHEMA_VERSION,
                "topic": "person",
                "stamp": round(now, 3),
                "present": False,
            }
            self._last = None

        if payload["present"] != self._was_present:
            rospy.loginfo("person %s", "ACQUIRED" if payload["present"] else "LOST")
            self._was_present = payload["present"]

        self.pub.publish(String(data=json.dumps(payload)))


if __name__ == "__main__":
    rospy.init_node("person_node")
    PersonNode()
    rospy.spin()
