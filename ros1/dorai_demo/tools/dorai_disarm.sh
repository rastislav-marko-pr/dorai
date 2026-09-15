#!/usr/bin/env bash
# dorai_disarm.sh — stop the robot being driven by the DORAI demo. Alias: `disarm`.
#
# Stops the armed behaviour node with SIGINT so its shutdown hook sends zero
# velocity. That matters: the local planner holds the last command for up to
# 2 s (last_cmd_vel_msg_time_max), so a node killed mid-approach would leave
# the robot walking. Then starts a motion-off behaviour node so the dashboard's
# state strip keeps working. Leaves the control mode alone: the robot stays
# standing. For a real stop use the physical e-stop, or `estop`.
# ROS setup files reference unset variables, so source them BEFORE set -u.
source /opt/ros/noetic/setup.bash
source "$HOME/ws/xmas_ws/devel/setup.bash"
set -u

LOG=/tmp/dorai_behaviour.log
forced=0

for pid in $(pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py"); do kill -INT "$pid" 2>/dev/null; done
for _ in $(seq 1 25); do pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py" >/dev/null || break; sleep 0.2; done
if pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py" >/dev/null; then
  for pid in $(pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py"); do kill -KILL "$pid" 2>/dev/null; done
  forced=1
fi

if [ "$forced" -eq 1 ]; then
  # The shutdown hook never ran, so the zeros were never sent. Send them here,
  # but only if the planner is actually listening.
  if rostopic info /robot_1/cmd_vel 2>/dev/null | grep -q 'local_planner'; then
    python3 - <<'EOF'
import rospy
from geometry_msgs.msg import Twist
rospy.init_node("dorai_disarm_zero", anonymous=True, disable_signals=True)
p = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=1)
rospy.sleep(0.5)
for _ in range(15):
    p.publish(Twist()); rospy.sleep(0.04)
EOF
  fi
fi

rosparam delete /behaviour_node >/dev/null 2>&1
setsid nohup rosrun dorai_demo behaviour_node.py _enable_motion:=false >"$LOG" 2>&1 </dev/null &
sleep 3

if rostopic info /robot_1/cmd_vel 2>/dev/null | grep -q '/behaviour_node'; then
  printf '\033[1;31m%s\033[0m\n' "WARNING: behaviour_node still publishes velocity. Use the e-stop."
  exit 1
fi
printf '\033[1;32m%s\033[0m\n' "DISARMED. Behaviour node running with motion off; robot left standing."
[ "$forced" -eq 1 ] && echo "  (node had to be force-killed; zero velocity was sent separately)"
exit 0
