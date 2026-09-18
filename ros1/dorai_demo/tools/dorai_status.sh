#!/usr/bin/env bash
# dorai_status.sh — one screen of what is and is not running. Alias: `demo-status`.
source /opt/ros/noetic/setup.bash
source "$HOME/ws/xmas_ws/devel/setup.bash"

ok()  { printf '  %-24s \033[1;32m%s\033[0m\n' "$1" "$2"; }
bad() { printf '  %-24s \033[1;31m%s\033[0m\n' "$1" "$2"; }
flows() { timeout "${2:-3}" rostopic echo -n1 "$1" >/dev/null 2>&1; }

echo "ROS 2 containers"
for c in voice doa wake stt bridge web; do
  s=$(docker ps --filter "name=dorai-$c-1" --format '{{.Status}}')
  [ -n "$s" ] && ok "$c" "$s" || bad "$c" "not running"
done

echo "ROS 1 side"
flows /atb_front/nn/spatial_detections/header/stamp 4 && ok camera "detections flowing" || bad camera "no detections"
flows /dorai/person && ok person "publishing" || bad person "not publishing"
flows /dorai/doa && ok "bearing into ROS 1" "flowing (bridge ok)" || bad "bearing into ROS 1" "none, check bridge"
if pgrep -f "^python3 [^ ]*/dorai_demo/behaviour_node\.py" >/dev/null; then
  if [ "$(rosparam get /behaviour_node/enable_motion 2>/dev/null)" = "true" ] || \
     pgrep -af "^python3 [^ ]*/dorai_demo/behaviour_node\.py" | grep -q "_enable_motion:=true"; then
    ok behaviour "ARMED"
  else
    ok behaviour "running, motion off"
  fi
else
  bad behaviour "not running"
fi

echo "Robot"
st=$(timeout 3 rostopic echo -n1 /robot_1/control/state 2>/dev/null | awk '/data:/ {print $2}')
case "$st" in
  1) ok "control mode" "1 READY (standing)";;
  "") bad "control mode" "none, quad_sdk not running";;
  *) bad "control mode" "$st (not READY)";;
esac
pubs=$(rostopic info /robot_1/cmd_vel 2>/dev/null | awk '/^Publishers:/{f=1;next} /^Subscribers:/{f=0} f && /\*/ {print $2}' | tr '\n' ' ')
[ -n "$pubs" ] && echo "  velocity publishers      $pubs" || echo "  velocity publishers      none"
echo "Dashboard: http://$(hostname -I | tr ' ' '\n' | grep '^192.168.50.' | head -1):8080"
