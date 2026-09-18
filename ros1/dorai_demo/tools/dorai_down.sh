#!/usr/bin/env bash
# dorai_down.sh — stop the DORAI demo. Alias on atb: `demo-down`.
#
#   demo-down        stop the ROS 1 side (camera, person, bridge, behaviour)
#   demo-down all    also stop the ROS 2 containers
#
# The behaviour node is stopped with SIGINT FIRST, before anything else, so its
# shutdown hook sends zero velocity if it was armed: the local planner holds the
# last command for up to 2 s. The armed node started by `arm` is not a child of
# roslaunch, so stopping roslaunch alone would leave it running.
# Leaves the robot's control mode alone; the walking stack is not touched.
source /opt/ros/noetic/setup.bash
source "$HOME/ws/xmas_ws/devel/setup.bash"
set -u

BEH="^python3 [^ ]*/dorai_demo/behaviour_node\.py"
LAUNCH="^/usr/bin/python3 [^ ]*/roslaunch dorai_demo demo.launch"

for pid in $(pgrep -f "$BEH"); do kill -INT "$pid" 2>/dev/null; done
for _ in $(seq 1 25); do pgrep -f "$BEH" >/dev/null || break; sleep 0.2; done
for pid in $(pgrep -f "$BEH"); do kill -KILL "$pid" 2>/dev/null; done

for pid in $(pgrep -f "$LAUNCH"); do kill -INT "$pid" 2>/dev/null; done
for _ in $(seq 1 75); do pgrep -f "$LAUNCH" >/dev/null || break; sleep 0.2; done
for pid in $(pgrep -f "$LAUNCH"); do kill -KILL "$pid" 2>/dev/null; done
# Camera nodelets can outlive a killed roslaunch and keep the OAK claimed.
for pid in $(pgrep -f "nodelet .*atb_front"); do kill -INT "$pid" 2>/dev/null; done
sleep 1
for pid in $(pgrep -f "nodelet .*atb_front"); do kill -KILL "$pid" 2>/dev/null; done

yes | rosnode cleanup >/dev/null 2>&1
rosparam delete /behaviour_node >/dev/null 2>&1
echo "ROS 1 side stopped."

if [ "${1:-}" = "all" ]; then
  (cd "$HOME/ws/dorai" && docker compose stop 2>&1 | grep -E "Stopped" | sed 's/^/  /')
  echo "ROS 2 containers stopped. They stay down across reboots until demo-up."
fi
