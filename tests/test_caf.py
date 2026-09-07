import io
import json
import math
import os
from pathlib import Path
import pty
import runpy
import select
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import caf_graphics as graphics
import caf_motion as motion

CAF = runpy.run_path(str(ROOT / "caf-art"), run_name="caf_test")
Scene = CAF["Scene"]


def scene():
    with patch.object(
        motion, "calibration_path", return_value=ROOT / ".build/no-calibration.json"
    ):
        s = Scene("diner")
    s.mode3d = True
    s.layout(90, 31)
    s._imu_flip = (1.0, 1.0, False)
    return s


def lean(s, axis=0, angle=15, seconds=2, dt=0.01):
    r, f, u = s._imu_basis
    b = (r, f)[axis]
    a = tuple(
        u[i] * math.cos(math.radians(angle)) - b[i] * math.sin(math.radians(angle))
        for i in range(3)
    )
    for _ in range(round(seconds / dt)):
        s.imu_input(a, (0, 0, 0), dt)
    return a


def level(s):
    for _ in range(200):
        s.imu_input(motion.unit((0.05, -0.34, -0.92)), (0, 0, 0), 0.01)


class MotionTests(unittest.TestCase):
    def test_calibration_axes_and_signs(self):
        a = math.radians(15)
        c = motion.calibrate(
            (0, 0, 1), (-math.sin(a), 0, math.cos(a)), (0, math.sin(a), math.cos(a))
        )
        self.assertAlmostEqual(c["right"][0], 1)
        self.assertAlmostEqual(c["near"][1], 1)
        # Repeat with all three sensor axes permuted: no sensor index assumptions.
        rotate = lambda v: (v[2], v[0], v[1])
        c2 = motion.calibrate(
            rotate((0, 0, 1)),
            rotate((-math.sin(a), 0, math.cos(a))),
            rotate((0, math.sin(a), math.cos(a))),
        )
        self.assertEqual(c2["right"], rotate(c["right"]))
        self.assertEqual(c2["near"], rotate(c["near"]))

    def test_reject_small_or_repeated_poses(self):
        with self.assertRaises(ValueError):
            motion.calibrate((0, 0, 1), (0, 0, 1), (0, 0.25, 0.97))
        with self.assertRaises(ValueError):
            motion.calibrate((0, 0, 1), (0.25, 0, 0.97), (0.25, 0, 0.97))

    def test_calibration_roundtrip_and_invalid(self):
        c = motion.calibrate((0, 0, 1), (-0.25, 0, 0.97), (0, 0.25, 0.97))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "imu.json"
            motion.save_calibration(c, p)
            self.assertEqual(motion.load_calibration(p), c)
            for value in ("null", "[]", "{", '{"version":1,"right":[0,0,0]}'):
                p.write_text(value)
                self.assertIsNone(motion.load_calibration(p))

    def test_both_tilt_axes_and_roll(self):
        for axis in (0, 1):
            s = scene()
            level(s)
            lean(s, axis)
            self.assertGreater(s.tilt_in[axis], 0.5)
            self.assertLess(abs(s.tilt_in[1 - axis]), 0.01)
            self.assertAlmostEqual(s.roll, s.tilt_in[0] * 0.09, places=3)
            self.assertIsNotNone(CAF["liquid_plane"](s, 0.8))

    def test_saved_axes_override_heuristic(self):
        s = scene()
        c = motion.calibrate((0, 0, 1), (0, -0.25, 0.97), (0.25, 0, 0.97))
        s._calibration = c
        for _ in range(200):
            s.imu_input((0, 0, 1), (0, 0, 0), 0.01)
        for _ in range(200):
            s.imu_input(motion.unit((0, -0.25, 0.97)), (0, 0, 0), 0.01)
        self.assertGreater(s.tilt_in[0], 0.5)
        self.assertLess(abs(s.tilt_in[1]), 0.01)

    def test_locked_pose_and_explicit_zero(self):
        s = scene()
        level(s)
        a = lean(s, seconds=90, dt=0.1)
        self.assertGreater(s.tilt_in[0], 0.5)
        s.imu_level()
        for _ in range(15):
            s.imu_input(a, (0, 0, 0), 0.1)
        self.assertLess(abs(s.tilt_in[0]), 0.001)
        self.assertLess(abs(s.roll), 0.001)

    def test_legacy_auto_level_opt_in(self):
        s = scene()
        s.auto_level = True
        level(s)
        lean(s, seconds=120, dt=0.1)
        self.assertLess(abs(s.tilt_in[0]), 0.08)

    def test_gyro_is_sample_rate_independent(self):
        kicks = []
        for dt in (0.01, 0.05):
            s = scene()
            level(s)
            a = tuple(s._imu_up)
            g = tuple(v * 4 for v in a)
            for _ in range(round(0.5 / dt)):
                s.imu_input(a, g, dt)
            kicks.append(s.yaw_kick)
        self.assertAlmostEqual(*kicks, places=5)

    def test_samples_buffer_unique_timestamps(self):
        samples = [(time.monotonic(), (0, 0, 1), (0, 0, 0), -1)]
        sampler = motion.SensorSamples(lambda: samples[-1])
        try:
            time.sleep(0.04)
            self.assertEqual(len(sampler.drain()), 1)
            time.sleep(0.02)
            self.assertEqual(sampler.drain(), [])
            samples.append((time.monotonic(), (0, 0.1, 0.99), (0, 0, 0), -1))
            time.sleep(0.03)
            self.assertEqual(len(sampler.drain()), 1)
        finally:
            sampler.close()
        self.assertFalse(sampler.thread.is_alive())

    def test_imu_rejects_nan_and_versions(self):
        read = CAF["imu_read_sample"]
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(read.__globals__, IMU_PATH=str(Path(d) / "sample")),
        ):
            p = Path(d) / "sample"
            for version, val in ((2, float("nan")), (99, 0.0)):
                p.write_bytes(
                    struct.pack(
                        CAF["IMU_FMT"],
                        b"CIMU",
                        version,
                        time.monotonic(),
                        val,
                        0,
                        1,
                        0,
                        0,
                        0,
                        -1,
                        val,
                    )
                )
                self.assertIsNone(read())


class GraphicsTests(unittest.TestCase):
    def test_probe_matches_only_our_response(self):
        data = b"s\x1b_Gi=73191;OK\x1b\\q"
        self.assertEqual(graphics.parse_probe(data), (True, b"sq"))
        self.assertEqual(
            graphics.parse_probe(b"\x1b_Gi=73191;ENOTSUP\x1b\\"), (False, b"")
        )
        other = b"\x1b_Gi=42;OK\x1b\\"
        self.assertEqual(graphics.parse_probe(other), (False, other))

    def test_non_tty_does_not_probe(self):
        self.assertEqual(
            graphics.probe_terminal(io.StringIO(), io.StringIO()), (False, "")
        )

    def test_dimensions_bounded_preserve_aspect(self):
        for cols, rows in ((26, 10), (90, 30), (300, 100), (1000, 200)):
            w, h = graphics.dimensions(cols, rows)
            self.assertLessEqual(w, 960)
            self.assertLessEqual(h, 720)
            self.assertLess(abs(w / h - cols / ((rows - 1) * 2)), 0.01)

    def test_png_roundtrip(self):
        rgb = bytes(range(18))
        png = graphics.rgb_png(3, 2, rgb)
        pos = 8
        chunks = {}
        while pos < len(png):
            (length,) = struct.unpack_from(">I", png, pos)
            kind = png[pos + 4 : pos + 8]
            data = png[pos + 8 : pos + 8 + length]
            (crc,) = struct.unpack_from(">I", png, pos + 8 + length)
            self.assertEqual(crc, zlib.crc32(kind + data))
            chunks[kind] = data
            pos += length + 12
        self.assertEqual(
            zlib.decompress(chunks[b"IDAT"]), b"\0" + rgb[:9] + b"\0" + rgb[9:]
        )
        with self.assertRaises(ValueError):
            graphics.rgb_png(3, 2, b"bad")

    def test_frame_contract(self):
        s = scene()
        s.tilt_in = (0.4, -0.3)
        s.roll = 0.04
        f = graphics.frame_payload(s, 540, 360)
        self.assertEqual(len(f["u"]), 36)
        self.assertEqual(len(f["water"]), 54 * 26)
        self.assertEqual(f["u"][6:9], [0.04, 0.4, -0.3])
        self.assertEqual(f["u"][27], 1.0)
        self.assertEqual(f["u"][29], len(f["dots"]) // 4)
        json.dumps(f, allow_nan=False)

    def test_camera_roll_basis_orthonormal(self):
        for roll in (-0.2, 0, 0.2):
            v = CAF["cam3d"](0.6, 0.4, roll)
            f, r, u = v[3:6], v[6:9], v[9:12]
            for a in (f, r, u):
                self.assertAlmostEqual(motion.dot(a, a), 1)
            for a, b in ((f, r), (f, u), (r, u)):
                self.assertAlmostEqual(motion.dot(a, b), 0)

    def test_hires_not_silently_halved(self):
        s = scene()
        s.S = 2
        s.layout(180, 59)
        px = CAF["render3d"](s)
        self.assertEqual((len(px[0]), len(px)), (180, 116))
        self.assertEqual(s._zb[1:], (180, 116))

    def test_odd_size_cpu_and_roll(self):
        s = scene()
        s.layout(151, 41)
        s.roll = 0.05
        px, grid = s.draw()
        self.assertEqual((len(px[0]), len(px)), (151, 80))
        lines, _ = s.blit(px, grid)
        self.assertEqual(len(lines), 40)

    def test_malformed_helper_exits_and_reaps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "caf.metal").write_text("unused")
            exe = p / "caf-metal"
            exe.write_text("#!/bin/sh\nprintf WRONG\n")
            exe.chmod(0o755)
            with self.assertRaises(OSError):
                graphics.MetalRenderer(p / "caf_graphics.py")

    def test_demo_ansi_no_graphics(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "caf-art"),
                "--demo",
                "1",
                "--size",
                "31x12",
                "--ansi",
            ],
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertNotIn(b"\x1b_G", proc.stdout)
        self.assertIn(b"\xe2\x96", proc.stdout)

    def test_2d_demo_still_works(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "caf-art"),
                "--demo",
                "2",
                "--size",
                "45x18",
                "--2d",
            ],
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())

    def test_cli_graphics_flags_are_not_reset_after_parsing(self):
        for flags, expected in (
            (
                [
                    "--renderer",
                    "metal",
                    "--fluid",
                    "legacy",
                    "--auto-level",
                    "--smooth",
                ],
                True,
            ),
            (["--renderer", "cpu", "--hires"], False),
            (["--renderer", "metal", "--ansi", "--hires"], False),
        ):
            rendered = []

            class FakeGPU:
                def render(self, s, w, h, dither):
                    rendered.append((s.real_fluid, s.auto_level, dither))
                    return bytes(w * h * 3)

                def close(self):
                    pass

            with (
                patch.dict(CAF["main"].__globals__, MetalRenderer=FakeGPU),
                patch.object(
                    sys, "argv", ["caf", "--demo", "1", "--size", "30x12", *flags]
                ),
                patch.object(sys, "stdout", io.StringIO()),
            ):
                CAF["main"]()
            self.assertEqual(bool(rendered), expected)
            if expected:
                self.assertEqual(rendered, [(False, True, False)])

    def test_terminal_probe_hires_quit_and_cleanup(self):
        # Execute the real loop behind a PTY; emulate only the terminal, never sensors.
        script = """import runpy,sys
m=runpy.run_path(sys.argv[1],run_name='test_pty')
g=m['main'].__globals__
g['imu_start_bridge']=lambda:None
g['imu_read_sample']=lambda:None
sys.argv=['caf','--renderer','cpu','--hires','--no-caffeinate','--size','30x12']
raise SystemExit(g['main']())
"""
        master, slave = pty.openpty()
        env = dict(os.environ, TERM="xterm-256color")
        proc = subprocess.Popen(
            [sys.executable, "-c", script, str(ROOT / "caf-art")],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=ROOT,
        )
        os.close(slave)
        data = b""
        responded = quit_sent = False
        deadline = time.monotonic() + 8
        try:
            while time.monotonic() < deadline and proc.poll() is None:
                if not select.select([master], [], [], 0.1)[0]:
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                data += chunk
                if b"a=q" in data and not responded:
                    os.write(master, b"\x1b_Gi=73191;OK\x1b\\")
                    responded = True
                if b"a=T" in data and not quit_sent:
                    os.write(master, b"q")
                    quit_sent = True
            proc.wait(timeout=2)
            # Drain cleanup bytes which may arrive with process exit.
            while select.select([master], [], [], 0.05)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                data += chunk
            self.assertEqual(proc.returncode, 0, data[-1000:])
            self.assertTrue(responded and quit_sent)
            self.assertIn(b"\x1b[?1049l", data)
            self.assertIn(b"a=d,d=I,i=99", data)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(master)


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires an accessible Metal device"
)
class MetalTests(unittest.TestCase):
    def test_real_frames_motion_materials_waves_resize_and_clock(self):
        gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")
        try:
            s = scene()
            s.level = s.target_level = 0.8
            s.ambient = 0.8
            a = gpu.render(s, 320, 200)
            self.assertEqual(len(a), 320 * 200 * 3)
            self.assertGreater(len(set(zip(a[0::3], a[1::3], a[2::3]))), 100)
            s.tilt_in = (0.65, 0)
            s.roll = 0.05
            b = gpu.render(s, 320, 200)
            self.assertNotEqual(a, b)
            s.tilt_in = (0, 0.65)
            self.assertNotEqual(b, gpu.render(s, 320, 200))
            s.tilt_in = (0, 0)
            s.roll = 0
            s.water.disturb(0.1, 0.1, 0.5, 0.3)
            self.assertNotEqual(a, gpu.render(s, 320, 200))
            s.write_word("10:24")
            for _ in range(30):
                s.t += 1 / 30
                s.step(1 / 30)
            self.assertEqual(len(gpu.render(s, 480, 320)), 480 * 320 * 3)
            s.theme_key = "cyber"
            s.theme = CAF["THEMES"]["cyber"]
            self.assertNotEqual(a, gpu.render(s, 320, 200))
        finally:
            gpu.close()
        self.assertIsNone(gpu.proc)


if __name__ == "__main__":
    unittest.main()
