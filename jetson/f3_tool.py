#!/usr/bin/env python3
"""Betaflight F3 bench tool (USB VCP).

  python3 f3_tool.py PORT info              # firmware, board, live attitude / gyro / battery
  python3 f3_tool.py PORT cli CMD [CMD..]   # run CLI commands, print the output (no save)
  python3 f3_tool.py PORT apply FILE        # run a file of CLI commands, then 'save' (the FC reboots)
  python3 f3_tool.py PORT acc               # accelerometer calibration - robot standing LEVEL and still
"""
import struct
import sys
import time

from msp import MSP, MSP_ANALOG, make_request, parse_response

MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_BOARD_INFO, MSP_BUILD_INFO = 1, 2, 3, 4, 5
MSP_ACC_CALIBRATION = 205


def raw(m, cmd, wait=0.1):
    m.ser.reset_input_buffer()
    m.ser.write(make_request(cmd))
    time.sleep(wait)
    return parse_response(m.ser.read(m.ser.in_waiting or 64), cmd)


def info(m):
    api = raw(m, MSP_API_VERSION)
    var = raw(m, MSP_FC_VARIANT)
    ver = raw(m, MSP_FC_VERSION)
    board = raw(m, MSP_BOARD_INFO)
    build = raw(m, MSP_BUILD_INFO)
    print("firmware : %s %s   (MSP API %s)" % (var.decode() if var else "?", ".".join(str(b) for b in ver) if ver else "?",
                                              "%d.%d" % (api[1], api[2]) if api else "?"))
    print("board    : %s" % (board[:4].decode(errors="replace") if board else "?"))
    print("build    : %s" % (build[:19].decode(errors="replace") if build else "?"))
    for _ in range(3):
        print("attitude %s  gyro %s  battery %s" % (m.attitude(), m.gyro(), m.analog()))
        time.sleep(0.2)


def cli(m, cmds, save=False):
    s = m.ser
    s.reset_input_buffer()
    s.write(b"#\r\n")
    time.sleep(0.5)
    out = s.read(s.in_waiting or 1).decode(errors="replace")
    if "CLI" not in out and "#" not in out:
        print("no CLI prompt:", repr(out[-200:]))
        return
    for c in cmds:
        s.write((c.strip() + "\r\n").encode())
        time.sleep(0.15 if not c.startswith(("diff", "dump", "status", "get")) else 1.0)
        print(s.read(s.in_waiting or 1).decode(errors="replace").replace("\r", ""), end="")
    if save:
        s.write(b"save\r\n")
        time.sleep(1.0)
        print(s.read(s.in_waiting or 1).decode(errors="replace").replace("\r", ""))
        print("saved - the flight controller reboots now")
    else:
        s.write(b"exit\r\n")     # leaves CLI (reboots without saving)
        time.sleep(0.5)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    m = None
    for _ in range(60):          # the port can be held briefly after a reboot (e.g. by ModemManager)
        try:
            m = MSP(sys.argv[1], timeout=0.2)
            break
        except Exception:
            time.sleep(1.0)
    if m is None:
        sys.exit("could not open %s" % sys.argv[1])
    what = sys.argv[2]
    if what == "info":
        info(m)
    elif what == "cli":
        cli(m, sys.argv[3:])
    elif what == "apply":
        lines = [l for l in open(sys.argv[3]).read().splitlines() if l.strip() and not l.strip().startswith("#")]
        cli(m, lines, save=True)
    elif what == "acc":
        print("calibrating - keep the robot level and still")
        m.ser.write(make_request(MSP_ACC_CALIBRATION))
        time.sleep(3.0)
        print("done; attitude now:", m.attitude())
    else:
        print(__doc__)
    m.close()


if __name__ == "__main__":
    main()
