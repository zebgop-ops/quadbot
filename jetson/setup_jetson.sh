#!/bin/bash
# One-time Jetson Nano setup (JetPack 4.6 / Ubuntu 18.04, Python 3.6). Run with sudo from this folder.
set -e
apt-get update
apt-get install -y python3-numpy python3-serial
# ModemManager probes every new USB serial port (the F3 and the servo adapter) for ~20 s after each
# plug-in/reboot, holding it busy and sending AT commands into it. The robot has no modem.
systemctl disable --now ModemManager 2>/dev/null || true
# serial ports and gamepad without root
U=${SUDO_USER:-$USER}
groupadd -f gpio
usermod -aG dialout,input,gpio "$U"
# FTDI USB-serial adapters default to a 16 ms latency timer, which wrecks a 50 Hz loop: force 1 ms.
cat > /etc/udev/rules.d/99-quadbot-serial.rules <<'RULES'
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
RULES
# neck/jaw servos on hardware PWM (pins 32/33): let the gpio group drive /sys/class/pwm
cat > /etc/udev/rules.d/99-quadbot-pwm.rules <<'RULES'
SUBSYSTEM=="pwm*", PROGRAM="/bin/sh -c 'chown -R root:gpio /sys/class/pwm && chmod -R 770 /sys/class/pwm; chown -R root:gpio /sys/devices/7000a000.pwm/pwm && chmod -R 770 /sys/devices/7000a000.pwm/pwm'"
RULES
udevadm control --reload-rules && udevadm trigger
# steady timing: MAXN, and clocks pinned at max on every boot. With the default schedutil governor the
# CPU clocks down during each 20 ms loop pause and the policy step takes ~8.5 ms instead of ~0.8 ms.
nvpmodel -m 0 || true
cat > /etc/systemd/system/jetson-clocks.service <<'UNIT'
[Unit]
Description=Pin Jetson clocks at max (quadbot control loop timing)
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now jetson-clocks.service || jetson_clocks || true
echo
echo "Serial devices (put the right ones into config.json: servo_port / f3_port):"
ls -l /dev/serial/by-id/ 2>/dev/null || echo "  none found - plug in the servo adapter and the F3 (USB)"
echo
echo "Neck/jaw PWM: run 'sudo /opt/nvidia/jetson-io/jetson-io.py' -> Configure 40-pin header ->"
echo "enable pwm0 (pin 32) and pwm2 (pin 33), save and reboot."
ls /sys/class/pwm/ 2>/dev/null
echo "Log out and back in for the dialout/input/gpio groups to apply."
