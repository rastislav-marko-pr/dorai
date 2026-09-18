#!/usr/bin/env bash
# dorai_up.sh — bring the whole DORAI demo up, motion OFF. Alias on atb: `demo-up`.
#
# Two halves, started differently:
#   * ROS 2 containers (voice, doa, wake, stt, bridge, web): docker compose.
#     They normally come back on their own after a reboot (restart policy),
#     so this is a no-op for them unless something stopped them.
#   * ROS 1 side (camera + detection, person_node, bridge_node, behaviour_node
#     with motion off): `roslaunch dorai_demo demo.launch`. Nothing starts this
#     after a reboot; this script does.
#
# Safe to run twice: it only starts what is not already running. It never arms
# motion; use `arm` for that once the robot is standing and the teleop closed.
source /opt/ros/noetic/setup.bash
source "$HOME/ws/xmas_ws/devel/setup.bash"
set -u

LOG=/tmp/dorai_demo.log
grn() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red() { printf '\033[1;31m%s\033[0m\n' "$*"; }

echo "ROS 2 containers..."
(cd "$HOME/ws/dorai" && docker compose up -d 2>&1 | grep -vE "Running|^$" | sed 's/^/  /')

if ! rostopic list >/dev/null 2>&1; then
  red "No ROS master. roscore.service should be running: sudo systemctl start roscore"
  exit 1
fi

if pgrep -f "^/usr/bin/python3 [^ ]*/roslaunch dorai_demo demo.launch" >/dev/null; then
  echo "ROS 1 side already running."
else
  echo "ROS 1 side (camera takes ~40 s to boot the OAK)..."
  # Stale private params from an earlier launch would silently win over code
  # defaults, including enable_motion. Start clean.
  rosparam delete /behaviour_node >/dev/null 2>&1
  rosparam delete /person_node >/dev/null 2>&1
  (cd /tmp && setsid nohup roslaunch dorai_demo demo.launch >"$LOG" 2>&1 </dev/null &)
fi

# Wait for the camera, bounded by wall-clock time rather than a loop count:
# each probe can itself take seconds while the OAK is booting.
deadline=$((SECONDS + 90))
while [ $SECONDS -lt $deadline ]; do
  timeout 2 rostopic echo -n1 /atb_front/nn/spatial_detections/header/stamp >/dev/null 2>&1 && break
  sleep 1
done

bash "$(dirname "$0")/dorai_status.sh"
