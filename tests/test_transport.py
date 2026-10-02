"""Frame transport: Kitty bytes, shm lifecycle, probe, sizing, pipelining."""

import base64
import io
import mmap
import os
from pathlib import Path
import pty
import re
import runpy
import sys
import threading
import tty
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import caf_graphics as graphics
import caf_motion as motion

CAF = runpy.run_path(str(ROOT / "caf-art"), run_name="caf_transport_test")
shm = graphics._posixshmem


def exists(name):
    try:
        os.close(shm.shm_open(name, os.O_RDONLY, 0))
        return True
    except FileNotFoundError:
        return False


def read_shm(name, size):
    fd = shm.shm_open(name, os.O_RDONLY, 0)
    try:
        with mmap.mmap(fd, os.fstat(fd).st_size, prot=mmap.PROT_READ) as m:
            return bytes(m[:size])
    finally:
        os.close(fd)


def idat(png):
    pos, out = 8, b""
    while pos < len(png):
        length = int.from_bytes(png[pos : pos + 4], "big")
        if png[pos + 4 : pos + 8] == b"IDAT":
            out += png[pos + 8 : pos + 8 + length]
        pos += length + 12
    return out


class KittyBytes(unittest.TestCase):
    def test_shm_command_is_exact(self):
        cmd = graphics.kitty_shm("/caf1.2", 4, 3, 80, 23)
        self.assertEqual(
            cmd,
            "\x1b[H\x1b_Ga=T,f=24,t=s,s=4,v=3,S=36,i=99,p=1,q=2,c=80,r=23;"
            + base64.standard_b64encode(b"/caf1.2").decode()
            + "\x1b\\",
        )

    def test_frame_dispatch(self):
        self.assertEqual(graphics.kitty_frame(None, 2, 2, 10, 5), "")
        rgb = bytes(range(12))
        raw = graphics.kitty_frame(rgb, 2, 2, 10, 5)
        boxed = graphics.kitty_frame(graphics.Frame(2, 2, rgb, None), 99, 99, 10, 5)
        self.assertEqual(raw, boxed)
        self.assertIn("a=T,f=100,i=99,p=1,q=2,c=10,r=5,m=0;", raw)
        named = graphics.kitty_frame(graphics.Frame(2, 2, None, "/x"), 0, 0, 10, 5)
        self.assertIn("t=s,s=2,v=2,S=12", named)

    def test_png_chunks_large_frames(self):
        rgb = bytes(range(256)) * (1280 * 854 * 3 // 256)
        rgb += bytes(1280 * 854 * 3 - len(rgb))
        b64 = graphics.kitty_png(graphics.rgb_png(1280, 854, rgb), 90, 30)
        payloads = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\", b64)
        self.assertGreater(len(payloads), 1)
        self.assertTrue(all(len(p) <= 4096 for _, p in payloads))
        self.assertTrue(payloads[-1][0].endswith("m=0"))
        self.assertTrue(all(c.endswith("m=1") for c, _ in payloads[:-1]))

    def test_parallel_deflate_is_one_valid_zlib_stream(self):
        raw = os.urandom(300_000) + bytes(900_000) + b"\x07" * 123_457
        data = graphics.zlib_level1(raw)
        self.assertEqual(zlib.decompress(data), raw)
        self.assertLess(len(data), len(zlib.compress(raw, 1)) * 1.01 + 64)
        small = b"abc" * 10
        self.assertEqual(graphics.zlib_level1(small), zlib.compress(small, 1))
        rgb = os.urandom(700 * 500 * 3)
        stride = 700 * 3
        self.assertEqual(
            zlib.decompress(idat(graphics.rgb_png(700, 500, rgb))),
            b"".join(b"\0" + rgb[i : i + stride] for i in range(0, len(rgb), stride)),
        )


class Sizing(unittest.TestCase):
    def test_native_cell_pixels_within_budget(self):
        self.assertEqual(
            graphics.dimensions(90, 31, (16, 32), graphics.SHM_LIMIT), (1440, 960)
        )
        self.assertEqual(graphics.dimensions(90, 31), (540, 360))  # 6x12 default

    def test_budgets_cap_and_preserve_aspect(self):
        for cell in ((7, 15), (16, 32), (24.5, 52)):
            for limit in (graphics.PNG_LIMIT, graphics.SHM_LIMIT):
                for cols, rows in ((26, 10), (120, 40), (400, 120)):
                    w, h = graphics.dimensions(cols, rows, cell, limit)
                    self.assertLessEqual(w, limit[0])
                    self.assertLessEqual(h, limit[1])
                    self.assertLessEqual(w * h, limit[2])
                    want = cols * cell[0] / ((rows - 1) * cell[1])
                    self.assertLess(abs(w / h - want) / want, 0.01)

    def test_cell_and_window_replies_parsed_and_consumed(self):
        data = b"a\x1b[6;34;17tb\x1b[4;986;1530t\x1b_Gi=73192;OK\x1b\\c"
        found, cell, window, pending = graphics.parse_replies(
            data, (graphics.PROBE_ID, graphics.SHM_PROBE_ID), cells=True
        )
        self.assertEqual((found, cell, window, pending),
                         ({graphics.SHM_PROBE_ID: True}, (17, 34), (1530, 986), b"abc"))
        # Without asking, CSI replies are not ours to eat.
        self.assertEqual(graphics.parse_probe(data), (False, data))

    def test_kernel_cell_size(self):
        master, slave = pty.openpty()
        try:
            import fcntl
            import struct
            import termios

            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 1600, 1020))
            self.assertEqual(graphics.cell_pixels(slave), (16, 34))
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
            self.assertIsNone(graphics.cell_pixels(slave))
        finally:
            os.close(master)
            os.close(slave)


@unittest.skipUnless(shm, "POSIX shared memory unavailable")
class SharedMemory(unittest.TestCase):
    def test_lifecycle_and_reaping(self):
        frames = graphics.SharedFrames(keep=2)
        try:
            names = [frames.create(100) for _ in range(4)]
            self.assertEqual(len(set(names)), 4)
            self.assertTrue(all(len(n.encode()) <= 30 and n[0] == "/" for n in names))
            fd = shm.shm_open(names[0], os.O_RDONLY, 0)
            self.assertEqual(os.fstat(fd).st_size % os.sysconf("SC_PAGE_SIZE"), 0)
            os.close(fd)
            frames.discard(names[3])
            self.assertFalse(exists(names[3]))
            # Terminal semantics: it reads then unlinks; reaping tolerates ENOENT.
            frames.handoff(names[0])
            shm.shm_unlink(names[0])
            frames.handoff(names[1])
            frames.handoff(names[2])  # keep=2: names[0] reaped (already gone)
            self.assertTrue(exists(names[1]) and exists(names[2]))
            leftover = frames.create(10)
            frames.handoff(leftover)  # reaps names[1], never read by a terminal
            self.assertFalse(exists(names[1]))
        finally:
            frames.close()
        self.assertFalse(any(exists(n) for n in (*names, leftover)))


class FakeTerminal(threading.Thread):
    """Answers like kitty: reads (and unlinks) t=s objects, reports cell size."""

    def __init__(self, master, cell=b"\x1b[6;32;16t", shm_ok=True):
        super().__init__(daemon=True)
        self.master, self.cell, self.shm_ok, self.seen = master, cell, shm_ok, b""

    def run(self):
        while b"i=73191," not in self.seen:
            self.seen += os.read(self.master, 4096)
        reply = b""
        m = re.search(rb"\x1b_Gi=73192,[^;]*;([^\x1b]*)", self.seen)
        if m:
            name = base64.b64decode(m[1]).decode()
            ok = self.shm_ok and read_shm(name, 3) is not None
            shm.shm_unlink(name)
            reply += b"\x1b_Gi=73192;" + (b"OK" if ok else b"ENOENT") + b"\x1b\\"
        if b"\x1b[16t" in self.seen:
            reply += self.cell
        reply += b"x\x1b_Gi=73191;OK\x1b\\"
        os.write(self.master, reply)


@unittest.skipUnless(shm, "POSIX shared memory unavailable")
class Probe(unittest.TestCase):
    def probe(self, env=None, **kw):
        master, slave = pty.openpty()
        tty.setraw(slave)
        term = FakeTerminal(master, **kw)
        term.start()
        stdin = os.fdopen(os.dup(slave), "rb", buffering=0)
        stdout = os.fdopen(os.dup(slave), "w")
        try:
            with patch.dict(os.environ, env or {}, clear=False):
                for k in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
                    if not env or k not in env:
                        os.environ.pop(k, None)
                result = graphics.probe_graphics(stdin, stdout, timeout=2)
            term.join(2)
            return result, term.seen
        finally:
            stdin.close()
            stdout.close()
            os.close(master)
            os.close(slave)

    def test_shm_and_cell_size_in_one_round_trip(self):
        result, seen = self.probe()
        self.assertEqual(result, graphics.Probe(True, True, (16, 32), "x"))
        self.assertLess(seen.index(b"i=73192"), seen.index(b"i=73191"))

    def test_shm_refusal_and_ssh_fall_back(self):
        result, _ = self.probe(shm_ok=False, cell=b"")
        self.assertEqual(result, graphics.Probe(True, False, graphics.DEFAULT_CELL, "x"))
        result, seen = self.probe(env={"SSH_TTY": "/dev/ttys009"})
        self.assertNotIn(b"t=s", seen)
        self.assertFalse(result.shm)
        self.assertTrue(result.images)

    def test_non_tty(self):
        self.assertEqual(
            graphics.probe_graphics(io.StringIO(), io.StringIO()),
            graphics.Probe(False, False, graphics.DEFAULT_CELL, ""),
        )


class Cli(unittest.TestCase):
    def test_transport_flag_reaches_renderer(self):
        seen = []

        class FakeGPU:
            def render(self, s, w, h, dither):
                seen.append((self.transport, self.pipeline, w, h))
                return bytes(w * h * 3)

            def close(self):
                pass

        for flags, env, want in (
            (["--transport", "shm"], {}, ("shm", "auto")),
            ([], {"CAF_TRANSPORT": "shm", "CAF_PIPELINE": "on"}, ("shm", "on")),
            ([], {}, ("png", "auto")),  # never shm without a terminal's yes
        ):
            seen.clear()
            with (
                patch.dict(CAF["main"].__globals__, MetalRenderer=FakeGPU),
                patch.dict(os.environ, env),
                patch.object(sys, "argv", ["caf", "--demo", "1", "--size", "30x12",
                                           "--renderer", "metal", *flags]),
                patch.object(sys, "stdout", io.StringIO()),
                patch.object(motion, "calibration_path",
                             return_value=ROOT / ".build/no-calibration.json"),
            ):
                CAF["main"]()
            self.assertEqual(seen[0][:2], want)
        with (
            patch.object(sys, "argv", ["caf", "--transport", "pigeon"]),
            patch.object(sys, "stderr", io.StringIO()),
        ):
            self.assertEqual(CAF["main"](), 2)


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires an accessible Metal device"
)
class MetalTransport(unittest.TestCase):
    def setUp(self):
        with patch.object(
            motion, "calibration_path", return_value=ROOT / ".build/no-calibration.json"
        ):
            self.s = CAF["Scene"]("diner")
        self.s.mode3d = True
        self.s.layout(90, 31)
        self.gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")

    def tearDown(self):
        self.gpu.close()

    def test_shm_frame_matches_pipe_frame(self):
        rgb = self.gpu.render(self.s, 320, 200, dt=0)
        self.gpu.transport = "shm"
        frame = self.gpu.render(self.s, 320, 200, dt=0)
        self.assertIsInstance(frame, graphics.Frame)
        self.assertEqual((frame.width, frame.height, frame.rgb), (320, 200, None))
        self.assertEqual(read_shm(frame.shm, len(rgb)), rgb)
        self.gpu.close()
        self.assertFalse(exists(frame.shm))  # close reaps unread frames

    def test_unmappable_shm_falls_back_to_inline_pixels(self):
        self.gpu.transport = "shm"
        self.gpu.frames = graphics.SharedFrames()
        self.gpu.frames.create = lambda size: "/caf-missing-object"
        rgb = self.gpu.render(self.s, 64, 48, dt=0)
        self.assertEqual(len(rgb), 64 * 48 * 3)
        self.assertEqual(self.gpu.transport, "png")

    def test_pipelined_frames_lag_one_and_drain(self):
        self.s.real_fluid = True
        self.gpu.transport = "shm"
        self.gpu.pipeline = "on"
        self.s.fluid_events = [{"kind": "cream"}]
        times = []
        first = self.gpu.render(self.s, 160, 100, dt=1 / 30)
        self.assertIsNone(first)
        self.assertEqual(self.s.fluid_events, [])  # sent once, never resent
        out = []
        for size in ((160, 100), (200, 120), (160, 100)):
            self.s.t += 1 / 30
            out.append(self.gpu.render(self.s, *size, dt=1 / 30))
            times.append(self.gpu.stats["simulated_time"])
        # Each call returns the previous request, at its own size.
        self.assertEqual([(f.width, f.height) for f in out],
                         [(160, 100), (160, 100), (200, 120)])
        self.assertAlmostEqual(times[1] - times[0], 1 / 30, places=4)
        self.assertGreater(self.gpu.stats["cream_emitted"], 0)
        # Leaving pipelining drains the in-flight frame and its shm object.
        pending = self.gpu._pending[0][4]
        self.gpu.pipeline = "off"
        frame = self.gpu.render(self.s, 160, 100, dt=0)
        self.assertEqual(frame.width, 160)
        self.assertFalse(exists(pending))
        self.assertFalse(self.gpu._pending)

    def test_adaptive_pipeline_hysteresis(self):
        self.gpu.pipeline = "auto"
        self.gpu.stats = {"gpu_ms": 20}
        for _ in range(20):
            self.gpu.note_frame(0.015)
        self.assertTrue(self.gpu._pipelined)
        self.gpu.stats = {"gpu_ms": 4}
        for _ in range(20):
            self.gpu.note_frame(0.006)
        self.assertFalse(self.gpu._pipelined)


if __name__ == "__main__":
    unittest.main()
