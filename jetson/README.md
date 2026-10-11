# Quadbot on the Jetson: running the trained policy

Runs the "cat" locomotion policy (trained in Isaac Lab, `~/quadbot/sim/runs/cat_final`) on the real
robot at 50 Hz: 12 × STS3215 on one bus, the Betaflight F3 for attitude/gyro/battery current, keyboard
or gamepad for commands. Python 3.6 + numpy + pyserial only; the network runs as plain numpy (≈0.3 ms).

| file | what |
|---|---|
| `run_policy.py` | the robot runtime: stand up → policy loop → sit down, safety stops, CSV logs |
| `calibrate.py` | guided calibration (servo scan, zero offsets, directions, F3 axes, lifted stand test) |
| `servo_tool.py` | bench tool: set IDs, centre servos at 2048 before fitting horns |
| `controller.py` | hardware-free core: conventions, numpy policy, observation, action filter |
| `hardware.py` | servo bus + F3 I/O, unit conversions, fault detection; `MockHardware` |
| `head.py` | neck + jaw DS3218s on hardware PWM (pins 32/33): slew-limited moves, level-gaze, chomp |
| `sts.py`, `msp.py` | Feetech STS protocol, Betaflight MSP v1 |
| `config.json` | ports, servo IDs, calibration, safety limits, command ranges |
| `models/` | `quadbot_cat_policy.npz` (numpy, default) and `.onnx` (if onnxruntime is installed) |
| `export_npz.py` | re-export after retraining (run where torch is installed) |

## 1. Hardware
- Servo bus: FE-URT-1 or Waveshare bus-servo adapter on USB, 1 Mbps, 12 V to the servos through the
  F3's current shunt (see the main README).
- F3: USB to the Jetson (MSP over the VCP). Firmware: **Betaflight 4.0.6, BETAFLIGHTF3** (the last
  release with F3 support; 3.1 reports the gyro in raw counts). Flash from the Jetson:
  `f3_tool.py PORT cli dfu` (3.1) or `bl` (4.x) → `dfu-util -a 0 -s 0x08000000:mass-erase:force:leave
  -D betaflight_4.0.6_BETAFLIGHTF3.bin` (bin = `objcopy -I ihex -O binary` of the release hex).
  Then `f3_tool.py PORT apply f3_robot_settings.txt` - most importantly `serial_update_rate_hz = 2000`
  (the default 100 Hz makes each MSP read wait up to 10 ms: 20 ms per control step instead of 1.2 ms).
  Finally, with the F3 mounted and the robot standing level: `f3_tool.py PORT acc`.
  ModemManager must be off (`setup_jetson.sh` does it): it grabs new USB serial ports after every reboot.
- Servo IDs: LF roll/pitch/knee = 1/2/3, RF = 4/5/6, LR = 7/8/9, RR = 10/11/12 (change in config.json).
- Neck + jaw DS3218: signal to header pin 32 (neck, pwmchip0/pwm0) and pin 33 (jaw, pwmchip0/pwm2),
  power from the separate 6 V BEC, grounds common. Enable the pins once with
  `sudo /opt/nvidia/jetson-io/jetson-io.py` (pwm0 + pwm2), then reboot.

## 2. Setup (once)
```
sudo ./setup_jetson.sh        # numpy, pyserial, serial/gamepad permissions, FTDI latency 1 ms, max clocks
```
Put the two `/dev/serial/by-id/...` paths it prints into `config.json` (`servo_port`, `f3_port`).

Assembly step 1 (before horns): one servo at a time, `python3 servo_tool.py PORT setid 1 <ID>`, then
`python3 servo_tool.py PORT center <ID>` and fit that joint's bracket with the leg straight down.

## 3. Calibrate (in this order)
```
python3 calibrate.py scan     # all 12 answer?
python3 calibrate.py zero     # robot on a stand, legs hanging straight: records each joint's zero
python3 calibrate.py dirs     # move each joint the way it prompts: records which way each servo counts
python3 calibrate.py imu      # level / nose down / right side down / rock + turn: F3 axis signs
python3 calibrate.py show     # torque off, live readout: +q, gravity and gyro signs make sense?
python3 calibrate.py stand    # robot LIFTED: eases to the stand pose, reports per-joint error
python3 calibrate.py head     # neck: trim until the head is level; jaw: until the mouth just closes;
                              # then a small test move checks each servo's direction
```
Sanity check with `show`: level robot → `g ≈ 0 0 -1`; nose down → `g x > 0`; right side down →
`g y < 0`; turning left → `w z > 0`.

## 4. Run
```
python3 run_policy.py --stand-only   # first time: stand up and hold, no policy
python3 run_policy.py                # walk: w/s fwd/back, a/d sideways, q/e turn, space stop, x sit+quit
python3 run_policy.py --on-stand     # on the test stand: legs run in the air, hip roll kept clear of the column
```
Head: `r`/`f` head up/down (10° steps), `g` level, `c` chomp. Gamepad (`/dev/input/js0`) takes over
once a stick is moved: left stick = drive, right stick X = turn, right stick Y = head, B = chomp,
any other button = stop.

The head keeps the gaze level: the neck counter-rotates the body's pitch from the IMU
(`head.stabilize`). While walking the neck is held within ±30° (`walk_neck_limit_deg`) because the
policy was trained with the head fixed at neutral; the full range (−55…+50°) is available when
standing. Moves are slew-limited (neck 90°/s, jaw 300°/s). The DS3218 is assumed to be the 180°
version (2000 µs over 180° → `us_per_deg` 11.11); for the 270° version set 7.41. Command limits: vx −0.5…0.8 m/s, vy ±0.3, turn ±2 rad/s (config.json), slew 1 m/s².

Every run writes `logs/run_*.csv`: commands, gyro, gravity, all joint angles/velocities, raw actions,
servo targets, loop time, battery volts/amps (F3 shunt), hottest servo — enough to compare with the
simulation and to measure cost of transport.

Safety (config.json → `safety`): torque off immediately if the body tilts past 50°, a servo misses
5 reads in a row, the F3 stops answering, a servo reaches 70 °C or the battery drops below 10.2 V.
Ctrl-C or `x` eases down to a sit (body ~100 mm) with the head returning to level, before torque off;
on a fault the legs drop torque and the head PWM stops (servos go limp). `--no-head` leaves them alone.

## What has been verified (without the robot)
- Unit tests (`python3 tests/test_protocol.py`): packet bytes, parsing, conventions, filter, head
  (neutral start, slew rate, walking limit, gaze stabilisation, range clamp, chomp).
- `python3 run_policy.py --mock`: full loop at 50.2 Hz, 0.3 ms per step on a desktop.
- Isaac Sim, through this exact code (`~/quadbot/sim/jetson_sim_check.py`): sim state turned into raw
  servo ticks (random horn offsets and directions), Betaflight-style attitude (0.1°) and gyro (1°/s),
  then converted and run by `controller.py`. Observations matched training within sensor resolution,
  servo targets within 4e-7 rad, and it walked 3.11 m of the commanded 3.2 m in 8 s.

## Not verified yet — check on the real robot
- STS speed units (assumed steps/s, 4096 per turn). If `show` gives odd joint velocities, set
  `"joint_velocity_source": "finite_difference"` in config.json.
- Betaflight's MSP_RAW_IMU gyro scale (assumed deg/s, as in BF 3.2+/4.x).
- Head PWM sysfs paths (pwmchip0 pwm0/pwm2 on JetPack 4.6; check `ls /sys/class/pwm` after jetson-io)
  and whether your DS3218s are the 180° or 270° version.
- Loop time on the Jetson (target < 10 ms of the 20 ms budget; shown live and logged).
- The servo model the policy was trained on (stiffness 12 N·m/rad, damping 0.25) vs the real servo's
  position loop; if the gait is jittery or sluggish, the STS P gain (EEPROM addr 21, default 32) is the knob.
