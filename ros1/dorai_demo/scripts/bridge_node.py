#!/usr/bin/env python3
"""
bridge_node: the ROS 1 half of the UDP link to the ROS 2 audio container.

  container -> ROS 1   127.0.0.1:5801   doa, wake      (we bind and receive)
  ROS 1 -> container   127.0.0.1:5802   person, robot_state (we send)

Deliberately a dumb pipe. It does not interpret bearings, pair wake words with
directions, or hold state. wake_mod owns the pairing; the behaviour node owns
the decisions. All this does is move bytes and enforce the contract's rate
limits.

Robustness. The other end is a container that may be down, restarting, or newer
than us. Three consequences, all handled:
  * A UDP send to a loopback port with nothing bound raises ECONNREFUSED, on
    THIS socket, on a LATER call, because the kernel queues the ICMP port
    unreachable. Naive code dies on the send after the one that was really
    refused. We swallow it and keep going.
  * Datagrams that are not UTF-8, not JSON, or not an object get dropped with a
    throttled warning. A malformed sender must not take the bridge down.
  * Unknown "topic" values are ignored rather than treated as an error, so the
    ROS 2 side can add topics before we know about them.
"""
import errno
import json
import socket
import threading

import rospy
from std_msgs.msg import String

SCHEMA_VERSION = 1
MAX_DATAGRAM = 8192


class BridgeNode(object):
    def __init__(self):
        self.rx_host = rospy.get_param("~rx_host", "127.0.0.1")
        self.rx_port = int(rospy.get_param("~rx_port", 5801))
        self.tx_host = rospy.get_param("~tx_host", "127.0.0.1")
        self.tx_port = int(rospy.get_param("~tx_port", 5802))

        # Contract rate limits, as minimum send intervals.
        #
        # The tolerance matters. A publisher aiming at exactly 10 Hz jitters
        # either side of a 0.100 s period, so a strict "reject anything under
        # 0.100 s" limiter throws away every slightly-early message and the
        # link settles at about 6 Hz instead of the 10 Hz the contract asks
        # for. Measured: 6.3 Hz before this tolerance, 10 Hz after. Allow a
        # 10 percent early arrival, which still caps a runaway publisher.
        self.rate_tolerance = float(rospy.get_param("~rate_tolerance", 0.9))
        self.min_interval = {
            "person": 1.0 / float(rospy.get_param("~person_rate_hz", 10.0)),
            "robot_state": 1.0 / float(rospy.get_param("~robot_state_rate_hz", 5.0)),
        }
        self._last_sent = {}

        self.pub_doa = rospy.Publisher("/dorai/doa", String, queue_size=10)
        self.pub_wake = rospy.Publisher("/dorai/wake", String, queue_size=10)
        self._inbound = {"doa": self.pub_doa, "wake": self.pub_wake}

        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx_sock.setblocking(False)

        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_sock.bind((self.rx_host, self.rx_port))
        self.rx_sock.settimeout(0.5)

        self.stats = {"rx_ok": 0, "rx_bad": 0, "tx_ok": 0, "tx_drop": 0,
                      "tx_rate_limited": 0}

        rospy.Subscriber("/dorai/person", String, self.on_outbound, queue_size=10)
        rospy.Subscriber("/dorai/robot_state", String, self.on_outbound, queue_size=10)

        self._stop = threading.Event()
        self._rx_thread = threading.Thread(target=self.rx_loop)
        self._rx_thread.daemon = True
        self._rx_thread.start()

        rospy.Timer(rospy.Duration(30.0), self.on_stats)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("bridge_node up: rx %s:%d -> /dorai/{doa,wake}, "
                      "/dorai/{person,robot_state} -> tx %s:%d",
                      self.rx_host, self.rx_port, self.tx_host, self.tx_port)

    # ---- inbound: UDP -> ROS ------------------------------------------------

    def rx_loop(self):
        while not self._stop.is_set() and not rospy.is_shutdown():
            try:
                data, _addr = self.rx_sock.recvfrom(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError as exc:
                # On shutdown the socket is closed under this thread, which
                # surfaces as EBADF. That is the normal way this loop ends, not
                # something to shout about in the logs on every exit.
                if self._stop.is_set() or exc.errno == errno.EBADF:
                    return
                # ECONNREFUSED can surface on a bound socket too if we ever
                # send from it. Never fatal.
                if exc.errno in (errno.ECONNREFUSED, errno.EINTR):
                    continue
                rospy.logwarn_throttle(5.0, "bridge rx socket error: %s", exc)
                continue

            try:
                payload = json.loads(data.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("datagram is not a JSON object")
                topic = payload.get("topic")
            except (UnicodeDecodeError, ValueError) as exc:
                self.stats["rx_bad"] += 1
                rospy.logwarn_throttle(5.0, "bridge dropped bad datagram: %s", exc)
                continue

            pub = self._inbound.get(topic)
            if pub is None:
                # Unknown topic is not an error, per the contract.
                rospy.logdebug("bridge ignoring topic %r", topic)
                continue

            if payload.get("v") != SCHEMA_VERSION:
                rospy.logwarn_throttle(
                    10.0, "bridge got schema v=%r on %s, expected %d; passing through",
                    payload.get("v"), topic, SCHEMA_VERSION)

            self.stats["rx_ok"] += 1
            # Republish the original text so nothing is lost in a re-encode.
            pub.publish(String(data=data.decode("utf-8")))

    # ---- outbound: ROS -> UDP ----------------------------------------------

    def on_outbound(self, msg):
        try:
            payload = json.loads(msg.data)
            topic = payload.get("topic")
        except ValueError:
            self.stats["tx_drop"] += 1
            rospy.logwarn_throttle(5.0, "bridge got non-JSON on an outbound topic")
            return

        now = rospy.Time.now().to_sec()
        interval = self.min_interval.get(topic)
        if interval is not None:
            last = self._last_sent.get(topic, 0.0)
            if (now - last) < interval * self.rate_tolerance:
                self.stats["tx_rate_limited"] += 1
                return
            self._last_sent[topic] = now

        raw = msg.data.encode("utf-8")
        if len(raw) > MAX_DATAGRAM:
            self.stats["tx_drop"] += 1
            rospy.logwarn_throttle(5.0, "bridge dropping oversize %s datagram (%d B)",
                                   topic, len(raw))
            return

        try:
            self.tx_sock.sendto(raw, (self.tx_host, self.tx_port))
            self.stats["tx_ok"] += 1
        except OSError as exc:
            # ECONNREFUSED: nothing listening yet, or the container bounced.
            # Expected during bring-up, must never be fatal.
            self.stats["tx_drop"] += 1
            if exc.errno == errno.ECONNREFUSED:
                rospy.logwarn_throttle(
                    10.0, "bridge: nothing listening on %s:%d yet (%s dropped)",
                    self.tx_host, self.tx_port, topic)
            else:
                rospy.logwarn_throttle(5.0, "bridge tx error: %s", exc)

    def on_stats(self, _evt):
        rospy.loginfo("bridge stats: %s", self.stats)

    def shutdown(self):
        self._stop.set()
        try:
            self.rx_sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    rospy.init_node("bridge_node")
    BridgeNode()
    rospy.spin()
