# Demo interface contract (epic dorai-phv)

Frozen 2026-09-09. Three workstreams build against this in parallel. Change it
only by agreement, because every other stream is coding against it.

## Topology

```
  ROS 2 Jazzy, dorai container, host networking        ROS 1 Noetic, atb
  ---------------------------------------------        ------------------
  ReSpeaker -> voice_mod -> /dorai_clean_audio
                 |                    |
                 |                    +-> stt_mod  -> /dorai_transcript
                 +-> doa_mod  -> /dorai/doa
                 +-> wake_mod -> /dorai/wake
                                      |
                                 bridge_mod  <==UDP==>  dorai_demo/bridge_node
                                      |                        |
                                 web_mod dashboard        behaviour_node -> /robot_1/cmd_vel
                                                          person_node <- /atb_front/nn/spatial_detections
```

No `ros1_bridge`. See bead `dorai-phv.15` for why.

## Wire protocol

Both endpoints speak UTF-8 JSON, one object per UDP datagram, no framing, no
newline. The container runs with `network_mode: host`, so both ends use
loopback.

| direction | destination |
|---|---|
| container to ROS 1 | `127.0.0.1:5801` |
| ROS 1 to container | `127.0.0.1:5802` |

Every datagram carries `v` (schema version, currently 1), `topic`, and `stamp`
(float seconds since the Unix epoch, taken at the *origin* of the data, not at
send time). Datagrams stay under 8 KB. Unknown `topic` values are ignored, not
an error. A receiver that cannot parse a datagram drops it and keeps running.

Rate limits: `doa` at 10 Hz, `person` at 10 Hz, `robot_state` at 5 Hz, `wake`
event-driven.

## Topics

The same JSON object is both the UDP payload and the ROS message body. On both
sides these are `std_msgs/String` carrying that JSON. This matches the existing
dorai convention of String and Float32MultiArray rather than custom messages,
and it keeps the bridge a dumb pipe.

### `/dorai/doa` — published in ROS 2 by `doa_mod`, mirrored into ROS 1

```json
{
  "v": 1,
  "topic": "doa",
  "stamp": 1788975325.966,
  "bearing_deg": 137.4,
  "confidence": 0.82,
  "valid": true,
  "spectrum": [0.01, 0.03, "... 36 floats, 10 degrees per bin, index 0 = 0 deg"]
}
```

`bearing_deg` is 0 to 360, measured counter-clockwise from the ReSpeaker MIC1
reference. `valid` is false when the time-difference-of-arrival closure check
fails on more than 70 percent of speech frames, or when no speech is present.
When `valid` is false, `bearing_deg` must still be present but consumers must
not trust it. `spectrum` is the pooled SRP-PHAT power, normalised to a 0 to 1
peak, and is optional: omit the key rather than sending nulls.

### `/dorai/wake` — published in ROS 2 by `wake_mod`, mirrored into ROS 1

```json
{
  "v": 1,
  "topic": "wake",
  "stamp": 1788975330.120,
  "keyword": "artaban",
  "bearing_deg": 137.4,
  "confidence": 0.82
}
```

`wake_mod` owns the pairing. It keeps a short ring of recent `/dorai/doa`
messages and attaches the bearing whose stamp is nearest the wake word's own
audio window. The bridge stays a dumb pipe and the behaviour node stays dumb.
If no confident bearing is available within the window, publish with
`confidence` 0.0 and let the consumer decide.

### `/dorai/person` — published in ROS 1 by `person_node`, mirrored into ROS 2

```json
{
  "v": 1,
  "topic": "person",
  "stamp": 1788975331.050,
  "present": true,
  "bearing_deg": -12.3,
  "distance_m": 3.42,
  "score": 0.87
}
```

`bearing_deg` here is signed, relative to the robot's own forward axis, positive
to the left. This is deliberately a different convention from the acoustic
bearing, because it is a different quantity: one is measured from the microphone
board's reference, the other from where the robot is pointing. Do not conflate
them. When `present` is false the other fields are omitted.

### `/dorai/robot_state` — published in ROS 1 by `behaviour_node`, mirrored into ROS 2

```json
{
  "v": 1,
  "topic": "robot_state",
  "stamp": 1788975331.200,
  "state": "approach",
  "mode": 1,
  "cmd": {"vx": 0.35, "vy": 0.0, "wz": -0.12},
  "target_bearing_deg": 137.4,
  "distance_m": 3.42
}
```

`state` is one of `idle`, `turn`, `acquire`, `approach`, `stopped`. `mode` is the
raw quad_sdk control mode integer as last seen on `/robot_1/control/state`.
`cmd` is what the behaviour node is actually publishing, before the local
planner's own scaling.

## Robot control facts, already verified

Do not rediscover these.

- Velocity goes to `/robot_1/cmd_vel` as `geometry_msgs/Twist`. Only `linear.x`,
  `linear.y` and `angular.z` survive; the local planner hard-zeroes the rest.
- Publish at 20 Hz or faster. After 2.0 s of silence the planner zeroes the
  command.
- The planner scales what you send: forward by 0.7, backward by 0.35, strafe by
  0.3, yaw by 1.0. A magnitude below 0.05 makes the robot stand rather than step.
- `/robot_1/control/mode` is `std_msgs/UInt8`. Publish 1 for READY, which is only
  reachable from 0 (SIT). Publish 4 for the soft stop. Confirm the transition on
  `/robot_1/control/state`, do not assume it.
- Verify commands are landing by watching `/robot_1/cmd_vel_smooth`.
- The walking stack is not running by default. Start it with
  `roslaunch quad_utils artaban.launch robot_type:=artaban_v03 terrain_input_type:=sliding_grid`.

## Camera facts, already verified

- Device mx id `14442C10013FE7D600`, runs as `/atb_front`.
- Colour 30 Hz at 1280x720, depth 28.7 Hz, depth already aligned to the colour
  frame and to the same optical frame id.
- With the network running, colour drops to about 18 Hz and detections publish
  at about 10 Hz.
- `/atb_front/nn/spatial_detections` is `vision_msgs/Detection3DArray`. Position
  is in metres in the camera optical frame: `z` forward, `x` right.
- Use the YOLOv4-tiny COCO blob, person is class id 0. The default MobileNet
  model reported a full-frame aeroplane at 0.98 confidence on an empty scene.
- Two driver traps, both found and fixed on 2026-09-09. Before the fix YOLO
  returned ZERO detections on every scene, which is easily mistaken for a
  correctly empty room. It was not.
  1. The colour preview defaults to 300x300 and the driver rescales it to the
     network's 416x416. Fed that way the network detects nothing at all. Set a
     native 416x416 preview with `nn_i_disable_resize`. MobileNet only appeared
     to work because its input is 300x300, so it never hit this path.
  2. The driver reads a YOLO confidence threshold only from a top-level
     `nn_config.confidence_threshold`. The shipped `yolo.json` nests it inside
     `NN_specific_metadata`, where nothing reads it, so any threshold set in the
     shipped file is silently ignored.
- After the fix, detections run at about 15 Hz with correct metric positions and
  correct left and right sign. Verified against real objects in the room.
- Person confidence floor is 0.6 and minimum distance 1.0 m. At a 0.5 floor an
  empty bench was reported as a person 43 percent of the time, always at 0.52 m.
- Working probe launches to copy from live at `/tmp/atb_person_test.launch` and
  `/tmp/atb_yolo_test.launch` on atb.

## Audio facts, already verified

- ReSpeaker 4 Mic Array UAC1.0, id `ArrayUAC10`, USB `2886:0018`.
- NEVER pin an ALSA card index. atb rebooted on 2026-09-09 and the ReSpeaker
  moved from card 1 to card 2, with the NVidia HDMI device taking the slot it
  had held. Anything pinning `card 1`, `hw:1,0` or a numeric device index will
  break on the next reboot, and it will break silently by capturing the wrong
  device. Resolve by name at startup, which is what `mic_name_filter` in
  voice_mod already does. If the named device is absent, log everything that
  WAS found and fail loudly rather than auto-picking the first card.
- Six channels, S16_LE, 16 kHz fixed. Channel map `FL FR FC LFE SL SR`.
  Channel 0 is the processed mix, 1 to 4 are the raw capsules, 5 is the playback
  reference. Direction of arrival uses channels 1 to 4.
- Circular array, 32 mm radius. Use the known-circle shortcut, not the
  walk-around calibration.
- One clock, so no per-microphone drift correction is needed.

## File ownership

Three streams, no shared files. Stay inside your own list.

| stream | owns |
|---|---|
| ROS 2 audio | `docker-compose.yml`, `Dockerfile`, `voice_mod/`, `stt_mod/`, new `doa_mod/`, new `wake_mod/`, new `bridge_mod/` |
| ROS 1 robot | new `ros1/dorai_demo/` in full |
| frontend | `web_mod/` only |

`DEMO_INTERFACES.md` is shared and read-only to all three.

## Deployment

The laptop repo at `/home/cdf/ws/dorai` is the source of truth and holds the
beads database. atb has its own clone at `~/ws/dorai` on branch
`dockerize-and-capture-fixes`, whose working tree content is currently identical
to the laptop's. The containers run from that clone.

Deploy by `rsync` of your own paths only. A backup of atb's tree taken before
this work exists at `~/dorai_backup_20260909_213511.tgz`.

The ROS 1 package deploys to `~/ws/dorai/ros1/dorai_demo` on atb and is reached
by a single symlink at `~/ws/xmas_ws/src/dorai_demo`. That symlink is the only
change made anywhere inside `xmas_ws`.
