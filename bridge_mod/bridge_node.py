#!/usr/bin/env python3
"""
bridge_mod / bridge_node.py — the ROS 2 half of the UDP bridge.

    ROS 2 (Jazzy, container)                    ROS 1 (Noetic, atb host)
    ------------------------                    ------------------------
    /dorai/doa   ---.                              .--> /dorai/doa
    /dorai/wake  ---+--> 127.0.0.1:5801 --------->-'    /dorai/wake
    /dorai/person <--.                             .--- person_node
    /dorai/robot_state <-- 127.0.0.1:5802 <-------'---- behaviour_node

WHY UDP AND NOT ros1_bridge
---------------------------
See bead dorai-phv.15. What matters here is the consequence: this node must
behave itself when the other end does not exist. It is written so that every
failure mode of the ROS 1 half -- not started yet, killed, restarted, mid-roslaunch,
sending garbage -- is a no-op on this side.

  * TX uses an UNCONNECTED datagram socket. That is deliberate: a *connected*
    UDP socket on Linux surfaces the ICMP port-unreachable from a dead peer as
    ECONNREFUSED on the NEXT send, so the bridge would start throwing the
    moment the ROS 1 side went down and would need its own recovery path. An
    unconnected socket simply drops into the void and keeps working the
    instant the peer comes back.
  * RX binds once and reads in a daemon thread with a timeout, so shutdown is
    prompt. A datagram that is not UTF-8, not JSON, not an object, or carries
    an unknown `topic` is DROPPED, not raised -- DEMO_INTERFACES: "Unknown
    topic values are ignored, not an error. A receiver that cannot parse a
    datagram drops it and keeps running."
  * If port 5802 is already bound (a second bridge, or a leftover process) the
    node says so and keeps running as a transmit-only bridge, because half a
    bridge is much better than a crash loop during a demo.

RATE LIMITS AND SIZE, per DEMO_INTERFACES
-----------------------------------------
doa 10 Hz, person 10 Hz, robot_state 5 Hz, wake event-driven. The limits are
enforced on BOTH directions here rather than trusted: doa_mod already caps
itself, but a bridge that forwards whatever it is handed is how a 200 Hz
debug publisher takes out the far end. Datagrams over 8 KB are dropped with a
throttled warning rather than fragmented -- an oversized `spectrum` is a bug
worth seeing, not something to paper over.

The payload is passed through BYTE FOR BYTE. Both sides carry the same JSON
object as a std_msgs/String body and as the UDP payload, so this node never
re-serialises and can never silently change a field. It validates only that
the body parses and names a topic it is allowed to forward.
"""

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


MAX_DATAGRAM = 8192          # DEMO_INTERFACES: "Datagrams stay under 8 KB"
SCHEMA_V = 1


class BridgeMod(Node):
    def __init__(self):
        super().__init__("bridge_mod")
        from rcl_interfaces.msg import ParameterDescriptor
        dyn = ParameterDescriptor(dynamic_typing=True)

        # network_mode: host, so both ends are on loopback.
        self.declare_parameter("tx_host", "127.0.0.1")
        self.declare_parameter("tx_port", 5801)
        self.declare_parameter("rx_bind", "127.0.0.1")
        self.declare_parameter("rx_port", 5802)

        self.declare_parameter("doa_topic", "/dorai/doa")
        self.declare_parameter("wake_topic", "/dorai/wake")
        self.declare_parameter("person_topic", "/dorai/person")
        self.declare_parameter("robot_state_topic", "/dorai/robot_state")

        # 0 disables the limit for that topic; wake is event-driven so it has
        # no limit by default.
        self.declare_parameter("doa_rate_hz", 10.0, dyn)
        self.declare_parameter("wake_rate_hz", 0.0, dyn)
        self.declare_parameter("person_rate_hz", 10.0, dyn)
        self.declare_parameter("robot_state_rate_hz", 5.0, dyn)

        gp = self.get_parameter
        self.tx_addr = (str(gp("tx_host").value), int(gp("tx_port").value))
        self.rx_addr = (str(gp("rx_bind").value), int(gp("rx_port").value))

        self._tx_limits = {
            "doa": float(gp("doa_rate_hz").value),
            "wake": float(gp("wake_rate_hz").value),
        }
        self._rx_limits = {
            "person": float(gp("person_rate_hz").value),
            "robot_state": float(gp("robot_state_rate_hz").value),
        }
        self._last_tx = {}
        self._last_rx = {}
        self._n_tx = 0
        self._n_tx_drop = 0
        self._n_rx = 0
        self._n_rx_drop = 0

        self._sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_tx.setblocking(False)

        self._pubs = {
            "person": self.create_publisher(
                String, gp("person_topic").value, 10),
            "robot_state": self.create_publisher(
                String, gp("robot_state_topic").value, 10),
        }

        self.create_subscription(
            String, gp("doa_topic").value,
            lambda m: self._tx(m, "doa"), 20)
        self.create_subscription(
            String, gp("wake_topic").value,
            lambda m: self._tx(m, "wake"), 20)

        self._running = True
        self._sock_rx = None
        try:
            self._sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock_rx.bind(self.rx_addr)
            self._sock_rx.settimeout(0.5)
            self._rx_thread = threading.Thread(
                target=self._rx_loop, name="bridge-rx", daemon=True)
            self._rx_thread.start()
        except OSError as exc:
            self._rx_thread = None
            self.get_logger().error(
                f"could not bind {self.rx_addr[0]}:{self.rx_addr[1]} ({exc}). "
                f"Running TRANSMIT-ONLY: /dorai/doa and /dorai/wake still "
                f"reach ROS 1, but nothing comes back. Another bridge_mod is "
                f"the usual cause.")

        self.create_timer(30.0, self._log_health)
        self.get_logger().info(
            f"bridge_mod: ROS2 -> udp {self.tx_addr[0]}:{self.tx_addr[1]}, "
            f"udp {self.rx_addr[0]}:{self.rx_addr[1]} -> ROS2 "
            f"({'listening' if self._sock_rx else 'DISABLED'})")

    # -- ROS 2 -> UDP --------------------------------------------------------
    def _tx(self, msg, topic):
        if self._limited(self._last_tx, topic, self._tx_limits.get(topic, 0.0)):
            self._n_tx_drop += 1
            return
        body = msg.data
        try:
            obj = json.loads(body)
            if not isinstance(obj, dict) or obj.get("topic") != topic:
                raise ValueError(f"body is not a {topic} object")
        except Exception as exc:
            self._n_tx_drop += 1
            self.get_logger().warning(
                f"dropping malformed {topic} message: {exc}",
                throttle_duration_sec=5.0)
            return
        payload = body.encode("utf-8")
        if len(payload) > MAX_DATAGRAM:
            self._n_tx_drop += 1
            self.get_logger().warning(
                f"{topic} datagram is {len(payload)} B, over the {MAX_DATAGRAM} B "
                f"contract limit; dropped. Shrink `spectrum` "
                f"(doa_mod -p spectrum_bins:=0 removes it).",
                throttle_duration_sec=5.0)
            return
        try:
            self._sock_tx.sendto(payload, self.tx_addr)
            self._n_tx += 1
        except OSError as exc:
            # The far end being absent is the normal state before roslaunch.
            self._n_tx_drop += 1
            self.get_logger().warning(
                f"udp send failed ({exc}); the ROS 1 half is probably not up. "
                f"Continuing.", throttle_duration_sec=30.0)

    # -- UDP -> ROS 2 --------------------------------------------------------
    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._sock_rx.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    return
                time.sleep(0.1)
                continue
            try:
                body = data.decode("utf-8")
                obj = json.loads(body)
                if not isinstance(obj, dict):
                    raise ValueError("not an object")
                topic = obj.get("topic")
            except Exception:
                self._n_rx_drop += 1
                self.get_logger().warning(
                    "dropped an unparseable datagram", throttle_duration_sec=5.0)
                continue
            pub = self._pubs.get(topic)
            if pub is None:
                # Unknown topic is ignored, not an error.
                self._n_rx_drop += 1
                self.get_logger().info(
                    f"ignoring datagram for unknown topic {topic!r}",
                    throttle_duration_sec=30.0)
                continue
            if self._limited(self._last_rx, topic,
                             self._rx_limits.get(topic, 0.0)):
                self._n_rx_drop += 1
                continue
            pub.publish(String(data=body))
            self._n_rx += 1

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _limited(state, topic, rate_hz, burst=2.0):
        """True when this message is above `rate_hz` and should be dropped.

        Token bucket, NOT a minimum-gap gate, and the difference is not
        academic. The gate this replaced dropped anything arriving sooner than
        1/rate after the last message it accepted. That punishes jitter rather
        than rate: doa_mod publishes at 8.95 Hz mean, comfortably under the
        10 Hz contract limit, but with inter-arrival times ranging 90 to 195 ms
        because each bearing falls out of an audio frame rather than a timer.
        Every frame landing 90 ms after its predecessor was discarded even
        though the average was never once over the limit, and because accepting
        a message resets the clock the loss compounds. Measured on atb: 8.95 Hz
        in, 6.47 Hz out, so 28 percent of bearings were thrown away for nothing.
        The dashboard trail went chunky and the behaviour node got a needlessly
        sparse target.

        A bucket holding `burst` tokens and refilling at `rate_hz` enforces the
        same long-run average while letting a well-behaved publisher's jitter
        through. A genuine flood still drains the bucket and gets clipped, which
        is the only thing the limit was ever there to do.

        The other direction never showed this because person_node publishes off
        a precise timer at 10.000 Hz with a 0.2 ms standard deviation, so it sat
        just inside the old gate's 1 ms tolerance by luck rather than design.
        """
        if rate_hz <= 0:
            return False
        now = time.monotonic()
        tokens, last = state.get(topic, (burst, now))
        tokens = min(burst, tokens + (now - last) * rate_hz)
        if tokens < 1.0:
            state[topic] = (tokens, now)
            return True
        state[topic] = (tokens - 1.0, now)
        return False

    def _log_health(self):
        self.get_logger().info(
            f"tx={self._n_tx} (dropped {self._n_tx_drop}) "
            f"rx={self._n_rx} (dropped {self._n_rx_drop})")

    def shutdown(self):
        self._running = False
        for s in (self._sock_rx, self._sock_tx):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        if self._rx_thread is not None and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = BridgeMod()
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
