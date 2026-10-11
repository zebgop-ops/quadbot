#!/usr/bin/env python3
"""STS3215 bench tool (assembly step 1: IDs and centring before the horns go on).

  python3 servo_tool.py PORT scan              # list servo IDs on the bus
  python3 servo_tool.py PORT setid OLD NEW     # connect ONE servo only (new ones ship as ID 1)
  python3 servo_tool.py PORT center ID [ID..]  # torque on, go to 2048 (fit the horn in this position)
  python3 servo_tool.py PORT read ID [ID..]    # position / voltage / temperature
  python3 servo_tool.py PORT off ID [ID..]     # torque off
"""
import sys
import time

from sts import ADDR_GOAL_POS, ADDR_POS, ADDR_TORQUE, STSBus


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    bus = STSBus(sys.argv[1], timeout=0.01)
    cmd, ids = sys.argv[2], [int(v) for v in sys.argv[3:]]
    if cmd == "scan":
        found = [i for i in range(0, 253) if bus.ping(i)]
        print("servos:", found or "none")
    elif cmd == "setid":
        old, new = ids
        print("ID %d -> %d: %s" % (old, new, "ok" if bus.set_id(old, new) else "FAILED"))
    elif cmd == "center":
        for i in ids:
            bus.write_u16(i, ADDR_GOAL_POS, 2048)
            bus.write_u8(i, ADDR_TORQUE, 1)
        time.sleep(1.0)
        for i in ids:
            print("ID %d at %s (target 2048)" % (i, bus.read_u16(i, ADDR_POS)))
    elif cmd == "read":
        for i in ids:
            h = bus.read_health(i)
            print("ID %d: pos %s  %s" % (i, bus.read_u16(i, ADDR_POS),
                                         "%.1f V %d C %.2f A" % h if h else "no reply"))
    elif cmd == "off":
        for i in ids:
            bus.write_u8(i, ADDR_TORQUE, 0)
    else:
        print(__doc__)
    bus.close()


if __name__ == "__main__":
    main()
