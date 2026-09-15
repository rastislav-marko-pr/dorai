#!/usr/bin/env bash
# dorai_arm.sh — arm the DORAI demo behaviour node. Alias on atb: `arm`.
#
# Refuses unless every pre-flight check passes, then replaces whichever
# behaviour_node is running (normally the motion-off one from demo.launch)
# with one that is allowed to move the robot. Undo with `disarm`.
#
# The checks are the ones that have actually bitten on the robot:
#   * robot not standing           -> nothing to command, and arming during
#                                     stand-up fights the controller
#   * teleop still on cmd_vel      -> pr_teleop publishes zeros at 500 Hz and
#                                     drowns our 25 Hz commands
#   * camera silently stalled      -> seen 2026-09-15: processes alive, zero
#                                     frames, dashboard just says "no person"
#   * no body state                -> the turn falls back to open loop
#   * no bearing                   -> nothing can trigger a turn
# ROS setup files reference unset variables, so source them BEFORE set -u.
source /opt/ros/noetic/setup.bash
source "$HOME/ws/xmas_ws/devel/setup.bash"
set -u

LOG=/tmp/dorai_behaviour.log
red() { printf '\033[1;31m%s\033[0m\n' "$*"; }
grn() { printf '\033[1;32m%s\033[0m\n' "$*"; }

failed=0
pass() { printf '  %-28s %s\n' "$1" "ok"; }
fail() { printf '  %-28s ' "$1"; red "FAIL: $2"; failed=1; }

echo "DORAI pre-flight"

state=$(timeout 3 rostopic echo -n1 /robot_1/control/state 2>/dev/null | awk '/data:/ {print $2}')
if [ "$state" = "1" ]; then pass "robot standing (READY)"
else fail "robot standing (READY)" "control state is '${state:-none}', stand it up with the controller"; fi

pubs=$(rostopic info /robot_1/cmd_vel 2>/dev/null | awk '/^Publishers:/{f=1;next} /^Subscribers:/{f=0} f && /\*/ {print $2}' | grep -v '^/behaviour_node$' | tr '\n' ' ')
if [ -z "$pubs" ]; then pass "velocity topic free"
else fail "velocity topic free" "also published by: ${pubs}- close the teleop"; fi

if timeout 4 rostopic echo -n1 /atb_front/nn/spatial_detections/header/stamp >/dev/null 2>&1; then pass "camera detections"
else fail "camera detections" "no detections, restart the camera: roslaunch dorai_demo demo.launch"; fi

if timeout 3 rostopic echo -n1 /robot_1/state/ground_truth/header/stamp >/dev/null 2>&1; then pass "body state (yaw feedback)"
else fail "body state (yaw feedback)" "no /robot_1/state/ground_truth, is quad_sdk running?"; fi

if timeout 3 rostopic echo -n1 /dorai/doa >/dev/null 2>&1; then pass "direction of arrival"
else fail "direction of arrival" "no /dorai/doa, check the dorai containers"; fi

if [ "$failed" -ne 0 ]; then
  red "NOT ARMED."
  exit 1
fi

# Replace the running behaviour node. SIGINT, so its shutdown hook runs.
for pid in $(pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py"); do kill -INT "$pid" 2>/dev/null; done
for _ in $(seq 1 25); do pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py" >/dev/null || break; sleep 0.2; done
for pid in $(pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py"); do kill -KILL "$pid" 2>/dev/null; done

# roslaunch leaves its private params on the server and they silently win
# over code defaults, including a stale enable_motion. Start clean.
rosparam delete /behaviour_node >/dev/null 2>&1
setsid nohup rosrun dorai_demo behaviour_node.py _enable_motion:=true >"$LOG" 2>&1 </dev/null &

for _ in $(seq 1 50); do
  rostopic info /robot_1/cmd_vel 2>/dev/null | grep -q '/behaviour_node' && break
  sleep 0.2
done
if ! rostopic info /robot_1/cmd_vel 2>/dev/null | grep -q '/behaviour_node'; then
  red "behaviour_node did not come up armed, see $LOG"
  exit 1
fi

grn "ARMED at demo speeds. Say the keyword to start."
echo "  disarm:     disarm"
echo "  soft stop:  estop     (physical e-stop is the real one)"
echo "  log:        tail -f $LOG"
