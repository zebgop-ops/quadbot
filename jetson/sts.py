"""Feetech STS3215 serial bus (SCS/STS protocol, little-endian). Needs only pyserial.

Packet:  FF FF ID LEN INSTR PARAMS... CHECKSUM      LEN = len(PARAMS) + 2
Status:  FF FF ID LEN ERROR PARAMS... CHECKSUM      CHECKSUM = ~(ID + LEN + ... ) & 0xFF
"""
import struct
import time

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

# STS3215 control table (SRAM unless noted)
ADDR_ID = 5            # EEPROM
ADDR_P_GAIN = 21       # EEPROM, default 32
ADDR_MODE = 33         # EEPROM, 0 = position servo
ADDR_TORQUE = 40
ADDR_ACC = 41
ADDR_GOAL_POS = 42     # u16
ADDR_GOAL_TIME = 44    # u16
ADDR_GOAL_SPEED = 46   # u16, 0 = max
ADDR_LOCK = 55         # 0 = EEPROM writable
ADDR_POS = 56          # u16, 0..4095 = 0..360 deg
ADDR_SPEED = 58        # sign-magnitude (bit 15), steps/s
ADDR_LOAD = 60
ADDR_VOLT = 62         # u8, 0.1 V
ADDR_TEMP = 63         # u8, deg C
ADDR_CURRENT = 69      # u16, 6.5 mA

TICKS_PER_REV = 4096


def checksum(body):
    return (~sum(body)) & 0xFF


def make_packet(sid, inst, params=b""):
    params = bytes(params)
    body = bytes([sid, len(params) + 2, inst]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


def sign_mag16(v):
    """STS speeds/loads: bit 15 is the sign."""
    return -(v & 0x7FFF) if v & 0x8000 else v


def parse_status(buf):
    """All valid status packets in buf -> list of (id, error, params)."""
    out, i, n = [], 0, len(buf)
    while i + 5 < n:
        if buf[i] != 0xFF or buf[i + 1] != 0xFF or buf[i + 2] == 0xFF:
            i += 1
            continue
        sid, ln = buf[i + 2], buf[i + 3]
        end = i + 3 + ln                  # index of checksum (LEN counts error + params + checksum)
        if ln < 2 or end >= n:
            break
        if checksum(buf[i + 2:end]) == buf[end]:
            out.append((sid, buf[i + 4], bytes(buf[i + 5:end])))
            i = end + 1
        else:
            i += 1
    return out


class STSBus(object):
    def __init__(self, port, baud=1000000, timeout=0.02):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.timeout = timeout
        self.errors = 0

    def close(self):
        self.ser.close()

    def _txrx(self, pkt, n_expected, timeout=None):
        """One transaction. An unplugged / failing adapter reads as 'no reply', so the callers'
        missed-read counting turns it into a clean HardwareFault instead of an exception."""
        try:
            self.ser.reset_input_buffer()
            self.ser.write(pkt)
            if n_expected <= 0:
                return b""
            self.ser.timeout = timeout or self.timeout
            return self.ser.read(n_expected)
        except Exception:
            return b""

    # --- single servo -------------------------------------------------------------------------
    def ping(self, sid):
        st = parse_status(self._txrx(make_packet(sid, INST_PING), 6))
        return any(s[0] == sid for s in st)

    def read(self, sid, addr, n):
        st = parse_status(self._txrx(make_packet(sid, INST_READ, [addr, n]), 6 + n))
        for s in st:
            if s[0] == sid and len(s[2]) == n:
                return s[2]
        self.errors += 1
        return None

    def write(self, sid, addr, data, ack=True):
        pkt = make_packet(sid, INST_WRITE, bytes([addr]) + bytes(data))
        if not ack or sid == BROADCAST_ID:
            self._txrx(pkt, 0)
            return True
        return any(s[0] == sid for s in parse_status(self._txrx(pkt, 6)))

    def read_u16(self, sid, addr):
        d = self.read(sid, addr, 2)
        return None if d is None else struct.unpack("<H", d)[0]

    def write_u8(self, sid, addr, v, ack=True):
        return self.write(sid, addr, [v & 0xFF], ack)

    def write_u16(self, sid, addr, v, ack=True):
        return self.write(sid, addr, struct.pack("<H", int(v) & 0xFFFF), ack)

    # --- many servos in one transaction ---------------------------------------------------------
    def sync_write(self, addr, data_by_id):
        """data_by_id: {id: bytes}, all the same length. No reply."""
        items = list(data_by_id.items())
        n = len(items[0][1])
        params = bytearray([addr, n])
        for sid, d in items:
            params.append(sid)
            params += bytes(d)
        self._txrx(make_packet(BROADCAST_ID, INST_SYNC_WRITE, params), 0)

    def sync_read(self, addr, n, ids):
        """-> {id: bytes or None}. Each servo answers with its own status packet, in order."""
        pkt = make_packet(BROADCAST_ID, INST_SYNC_READ, bytes([addr, n]) + bytes(ids))
        raw = self._txrx(pkt, len(ids) * (6 + n), timeout=self.timeout + 0.0002 * len(ids) * (6 + n))
        got = {s[0]: s[2] for s in parse_status(raw) if len(s[2]) == n}
        out = {}
        for sid in ids:
            out[sid] = got.get(sid)
            if out[sid] is None:
                self.errors += 1
        return out

    def sync_write_positions(self, ticks_by_id):
        self.sync_write(ADDR_GOAL_POS, {sid: struct.pack("<H", max(0, min(4095, int(round(t)))))
                                        for sid, t in ticks_by_id.items()})

    def sync_read_pos_speed(self, ids):
        """-> {id: (position ticks, speed ticks/s) or None}"""
        out = {}
        for sid, d in self.sync_read(ADDR_POS, 4, ids).items():
            if d is None:
                out[sid] = None
            else:
                p, v = struct.unpack("<HH", d)
                out[sid] = (p, sign_mag16(v))
        return out

    def torque(self, ids, on):
        self.sync_write(ADDR_TORQUE, {sid: bytes([1 if on else 0]) for sid in ids})

    def read_health(self, sid):
        """-> (volts, temp C, current A) or None, one servo."""
        d = self.read(sid, ADDR_VOLT, ADDR_CURRENT + 2 - ADDR_VOLT)
        if d is None:
            return None
        cur = struct.unpack("<H", d[ADDR_CURRENT - ADDR_VOLT:ADDR_CURRENT - ADDR_VOLT + 2])[0]
        return d[0] * 0.1, d[1], sign_mag16(cur) * 0.0065

    def set_id(self, old, new):
        """EEPROM: unlock, write ID, lock (under the new ID)."""
        self.write_u8(old, ADDR_LOCK, 0)
        self.write_u8(old, ADDR_ID, new)
        time.sleep(0.05)
        self.write_u8(new, ADDR_LOCK, 1)
        return self.ping(new)
