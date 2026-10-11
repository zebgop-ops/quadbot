"""Betaflight MSP v1 client (F3 over USB VCP): attitude, gyro and battery current."""
import struct

MSP_RAW_IMU = 102
MSP_ATTITUDE = 108
MSP_ANALOG = 110


def make_request(cmd, payload=b""):
    payload = bytes(payload)
    c = len(payload) ^ cmd
    for b in payload:
        c ^= b
    return b"$M<" + bytes([len(payload), cmd]) + payload + bytes([c])


def parse_response(buf, cmd):
    """First '$M>' frame for cmd in buf -> payload bytes, or None."""
    i = 0
    while True:
        i = buf.find(b"$M", i)
        if i < 0 or i + 5 > len(buf):
            return None
        if buf[i + 2:i + 3] != b">":
            i += 2
            continue
        size, c = buf[i + 3], buf[i + 4]
        end = i + 5 + size
        if end >= len(buf):
            return None
        chk = size ^ c
        for b in buf[i + 5:end]:
            chk ^= b
        if c == cmd and chk == buf[end]:
            return bytes(buf[i + 5:end])
        i += 2


class MSP(object):
    def __init__(self, port, baud=115200, timeout=0.02):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.errors = 0

    def close(self):
        self.ser.close()

    def request(self, cmd, n_payload):
        """Send cmd; read the reply using its own size byte (Betaflight versions differ in
        payload length). Returns the payload if it has at least n_payload bytes."""
        try:
            self.ser.reset_input_buffer()
            self.ser.write(make_request(cmd))
            head = self.ser.read(5)
        except (OSError, IOError, Exception) as e:     # unplugged / rebooting: a missed read
            self.errors += 1
            return None
        i = head.find(b"$M")
        if i > 0:                                   # resync if junk came first
            head = head[i:] + self.ser.read(i)
        p = None
        if len(head) == 5 and head[:3] == b"$M>":
            try:
                p = parse_response(head + self.ser.read(head[3] + 1), cmd)
            except (OSError, IOError, Exception):
                p = None
        if p is None or len(p) < n_payload:
            self.errors += 1
            return None
        return p

    def attitude(self):
        """-> (roll, pitch, yaw) degrees as Betaflight reports them (signs fixed by calibration)."""
        p = self.request(MSP_ATTITUDE, 6)
        if p is None:
            return None
        r, pch, y = struct.unpack("<hhh", p[:6])
        return r / 10.0, pch / 10.0, float(y)

    def gyro(self):
        """-> gyro (x, y, z) deg/s in Betaflight's axes."""
        p = self.request(MSP_RAW_IMU, 18)
        if p is None:
            return None
        v = struct.unpack("<9h", p[:18])
        return float(v[3]), float(v[4]), float(v[5])

    def analog(self):
        """-> (battery volts, amps, mAh used)."""
        p = self.request(MSP_ANALOG, 7)
        if p is None:
            return None
        vbat, mah, _rssi, amps = struct.unpack("<BHHh", p[:7])
        return vbat / 10.0, amps / 100.0, mah
