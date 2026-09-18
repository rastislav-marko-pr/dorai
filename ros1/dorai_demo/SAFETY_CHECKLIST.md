# DORAI demo: safety and pre-flight checklist

Artaban is a real quadruped that walks towards a person on purpose. Everything
below is either a thing that has actually gone wrong here, or a thing whose
failure makes the robot move when nobody expects it to. Work through it in
order, out loud, with the room listening.

---

## 0. The stop, before anything else

**The soft stop is publishing `4` to `/robot_1/control/mode`.**

```bash
rostopic pub -1 /robot_1/control/mode std_msgs/UInt8 "data: 4"
```

Put that command in a terminal, on screen, before the demo starts. Do not type
it from memory while the robot is walking.

- **Name the person holding the stop, out loud, before the robot is armed.**
  One named person, not "whoever is nearest". They hold the physical e-stop and
  they do not also operate the laptop, run the dashboard, or talk to the
  audience. Their only job is the stop.
- Everyone in the room must know that mode `4` is the soft stop and where the
  physical e-stop is. Say it to the audience too, not just the team.
- The soft stop is *soft*. The physical e-stop is the real one. If the robot is
  doing something genuinely wrong, use the physical one and ask questions after.

`behaviour_node` never publishes to `/robot_1/control/mode`. Mode changes are a
human action, deliberately, so that nothing in this package can arm the robot or
paper over a fault by cycling the mode.

---

## 1. Motion is off until someone turns it on

`behaviour_node` defaults to `~enable_motion:=false`. In that state it runs the
whole state machine and publishes `/dorai/robot_state`, but it does not even
create the `/robot_1/cmd_vel` publisher. Confirm which state you are in:

```bash
rosnode info /behaviour_node | sed -n '/Publications/,/Subscriptions/p'
```

If `/robot_1/cmd_vel` is not listed, the node physically cannot command motion.
Arm it only when the rest of this checklist is green:

```bash
roslaunch dorai_demo demo.launch enable_motion:=true
```

To rehearse the behaviour safely with motion "on" but going nowhere, remap the
command onto a dead topic. This is how the state machine was tested:

```bash
rosrun dorai_demo behaviour_node.py _enable_motion:=true \
  _control_state_topic:=/dorai_test/control_state \
  /robot_1/cmd_vel:=/dorai_test/cmd_vel
```

---

## 2. ReSpeaker still on card 1

**This has already bitten. As of the last check the ReSpeaker was not connected
to atb at all** (no `2886:0018` on USB, no `ArrayUAC10`, and `/proc/asound/cards`
listed only the internal audio and the NVidia HDMI device). Do not assume it is
there.

```bash
lsusb | grep 2886:0018        # expect the ReSpeaker
cat /proc/asound/cards        # expect card 1 = ArrayUAC10
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 6 -d 2 /tmp/mic_check.wav
```

The card *number* is not stable. Plugging in any other USB audio device, or a
different boot order, renumbers the cards, and the audio container then happily
captures two seconds of silence from the wrong device. Check the number on the
day, and if it is not 1, fix the configuration rather than hoping.

Six channels, S16_LE, 16 kHz. Channel 0 is the processed mix, 1 to 4 are the raw
capsules used for direction of arrival, 5 is the playback reference.

---

## 3. Camera mx id matches, and is on USB 3

```bash
rosparam get /atb_front/camera_i_mx_id     # expect 14442C10013FE7D600
lsusb | grep 03e7                          # expect the Movidius VPU
```

**After a reboot the OAK came up on a USB 2.0 bus** (`03e7:f63c` on bus 003)
rather than USB 3 (`03e7:f63b` on bus 004). It still works, but the bandwidth is
much lower, which shows up as reduced colour frame rate and slower detections.
If the frame rate looks wrong, reseat the cable and confirm the bus before
blaming the software.

Confirm detections are actually flowing, not merely that the topic exists:

```bash
rostopic hz /atb_front/nn/spatial_detections   # expect roughly 10 Hz
rostopic echo -n1 /dorai/person
```

**An empty `Detection3DArray` publishes at the same rate as a full one.** A
healthy topic rate proves the camera is alive; it proves nothing whatsoever
about whether anybody is being detected. Section 7 matters more than this one.

---

## 4. The robot actually reached READY

Mode 1 is READY, and it is only reachable from mode 0 (SIT). Publishing 1 is a
request, not an outcome:

```bash
rostopic echo /robot_1/control/state
```

Confirm the transition on `/robot_1/control/state` before arming motion. Never
assume the mode took. `behaviour_node` gates all motion on having actually seen
READY (`~require_ready_mode`, default true) and will hold still and warn if it
has not, which is a safety net, not a substitute for looking.

Check the walking stack is up and commands are landing:

```bash
roslaunch quad_utils artaban.launch robot_type:=artaban_v03 \
          terrain_input_type:=sliding_grid
rostopic hz /robot_1/cmd_vel_smooth
```

`/robot_1/cmd_vel_smooth` is how you tell that commands are reaching the
planner. It is the planner echoing our own command back, so it confirms
transport only. It is **not** evidence that the robot moved.

---

## 4b. How long the robot takes to stop, measured

If the person disappears mid-approach the robot does not stop instantly. Two
delays stack: `person_node` holds a target across a dropped frame (~0.6 s), and
`behaviour_node` then waits for its own lost timeout (0.4 s) before commanding
zero.

Measured on the bench with motion remapped to a dead topic: the last non-zero
command came **0.38 s** after the target stopped being published, and the robot
was commanded to a full stop after that. Worst case from the camera genuinely
last seeing somebody is therefore about **1.0 s**, which at the demo's commanded
0.5 forward (about 0.35 m/s after the planner's 0.7 scale) is roughly **0.35 m
of extra travel**.

Budget that into the standoff and into how close the audience stands. It is also
why the numbers in `person_node.py` and `behaviour_node.py` are commented as
stacking: shortening one without knowing about the other gives a false sense of
how quickly the robot gives up.

## 5. Clear floor in the approach direction

The robot drives forward until the person is at the standoff distance, currently
**1.1 m** (`~standoff_m`, set in demo.launch), and it turns first towards
wherever the wake word came from. That direction is not known in advance.

- Clear a corridor at least **4 m long and 2 m wide** in every direction the
  robot might turn towards. It stops 1.1 m from the person, so a person standing
  3 m away means 1.9 m of actual travel, plus overshoot.
- **Expect it to end nearer than 1.1 m.** The measured stop latency is about 0.78 s, and at the demo
  approach speed that carries roughly a further 0.11 m, so expect the robot to
  come to rest around 1.0 m from the target person. Brief them on that number
  before they stand there.
- No cables, no bags, no laptop on the floor, no chair legs in the path.
- Audience stands behind the robot or well to the side, never in the approach
  corridor next to the target person.
- The person being approached should know they are the target, and should stand
  still. Someone walking towards the robot while it walks towards them halves
  the stopping distance.
- Nothing at head height for the robot to clip, and no glass doors, which the
  depth camera sees badly.

---

## 6. Bearing calibration

`~mic_forward_deg` is the offset between the robot's forward axis and the
ReSpeaker's 0 degree reference. It is the single calibration between hearing and
heading, it defaults to **270.0** (measured 2026-09-10, board 0 points to the
robot's left), and if it is wrong the robot confidently turns the wrong way. Measure it on the day: stand at a known angle, speak, and compare
`/dorai/doa` against where you actually are.

The acoustic bearing (0 to 360, counter-clockwise, from the microphone board)
and the camera bearing (signed, positive left, from the robot's forward axis)
are different quantities in different frames. Do not conflate them.

---

## 6a. What a failed turn now does

The turn is closed on the yaw measured on `/robot_1/state/ground_truth`, not on
the yaw rate the node commanded. Two consequences worth knowing before you watch
a run:

- A turn that hits its 12 s timeout with more than 20 degrees still outstanding
  is logged at **ERROR** as `TURN FAILED` and goes straight to `stopped`. It no
  longer walks on into `acquire`, which used to send the robot at whoever
  happened to be in front of it rather than whoever spoke. If you see that line,
  the robot did not turn; find out why before running again.
- If `/robot_1/state/ground_truth` stops, the node warns
  (`body state stale ... falling back to open-loop dead reckoning`) and reverts
  to the old commanded-value integral for as long as the topic is missing. It
  keeps moving. If you would rather it did not, stop the run by hand.
- `use_yaw_feedback:=false` and `use_odom_distance:=false` on the roslaunch line
  force the old open-loop behaviour back, with no edit and no redeploy.

---

## 7. Known open risk: no human has been detected yet

Read this before promising anybody a demo.

### What was fixed, and must not be undone

YOLOv4-tiny returned **zero detections through the ROS driver**, at every
confidence threshold down to 0.1, on a scene full of COCO objects. Two causes,
both now handled inside `dorai_demo` without touching any shipped package:

1. **The preview size.** The driver's colour preview defaults to 300x300 and it
   then rescales to the network's 416x416 through an ImageManip. Fed that way
   the network detects nothing at all. `detection_override.launch` now sets the
   preview to a native 416x416 and disables the rescale. MobileNet appeared to
   work only because its input is 300x300, so it never hit that path.
2. **The confidence threshold was never applied.** `depthai_ros_driver` reads a
   YOLO threshold only from a top-level `nn_config.confidence_threshold`, while
   the shipped `yolo.json` nests it inside `NN_specific_metadata` where nothing
   reads it. The package now ships `config/yolov4_tiny_coco.json` with the value
   in the place the driver actually looks.

After the fix, verified on the robot: `pottedplant` at 0.64, `vase`, `bicycle`,
with correct metric positions and the correct left/right sign. A hand-built
depthai pipeline on the same device detects people at up to 0.99, which is what
proves the blob itself was never the problem.

**If detections stop, check the preview size first.** Anyone "tidying"
`rgb_i_preview_*` or `nn_i_disable_resize` out of the launch file will silently
break detection completely while leaving every topic rate looking healthy.

### False positives, measured

On an **empty** workbench, with a 0.5 confidence floor, `/dorai/person` reported
somebody present **43 percent of the time**, scoring 0.50 to 0.73, always at a
fixed 0.52 m. That is clutter close to the lens, not a person.

Defaults are now `min_score` 0.6 and `min_distance_m` 1.0, which measured **0
false positives over 450 samples** on the same scene. Nobody the robot should
walk towards is closer than a metre and the standoff is 1.5 m, so the distance
floor costs nothing. Re-measure in the demo room; different clutter, different
false positives.

### Still unverified, and only a human can close it

**Nobody has yet stood in front of this camera and been detected as a person.**
The camera points at a workbench about a metre away, so no passer-by ever enters
the frame and passive sampling can never confirm it. Every piece either side of
that step is verified; this one step is not.

Before the demo, one person must stand at 2, 3 and 4 m and confirm that

```bash
rostopic echo /dorai/person
```

shows `present: true`, a `distance_m` that matches a tape measure, and a
`bearing_deg` whose sign matches the side they are standing on (positive is
left). Also walk across the frame to check the bearing sign flips the right way.

If that has not been done, the demo does not work, whatever the topic rates say.
A healthy rate on `/atb_front/nn/spatial_detections` proves only that the camera
is alive: an empty detection array publishes at exactly the same rate as a full
one.

If you ever switch back to MobileNet, person is class **15** there, not 0. Set
`person_node`'s `~person_class_id` accordingly or it will never see anybody.

## 8. Order of operations on the day

1. Physical e-stop in a named hand. Soft-stop command on screen.
2. Floor cleared, audience placed, target person briefed.
3. ReSpeaker present, card number confirmed, two-second recording checked.
4. Camera up, mx id confirmed, **a human confirmed as detected at 2 to 4 m**.
5. Audio container up, `/dorai/doa` sane for a known direction.
6. Bridge up, `/dorai/person` and `/dorai/robot_state` arriving in the container.
7. Robot to SIT, then READY, transition confirmed on `/robot_1/control/state`.
8. Walking stack up, `/robot_1/cmd_vel_smooth` alive.
9. Behaviour node **still unarmed**. Dry run one full wake-to-approach cycle and
   watch the states in `/dorai/robot_state`.
10. Only now, arm motion. Announce it. Then wake the robot.

If any step is red, stop and fix it. A demo that fails gracefully is recoverable;
a quadruped walking at an audience because somebody skipped step 4 is not.
