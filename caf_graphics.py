"""Optional Metal renderer and a conservative Kitty capability probe.

No third-party Python dependencies. Importing this module starts no processes.
"""

import json
import math
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import time
import zlib

PROBE_ID = 73191
REPLY = re.compile(rb"\x1b_G([^;]*);([^\x1b]*)\x1b\\")


def parse_probe(data):
    """Consume only our graphics reply; preserve keys and unrelated replies."""
    supported = False

    def consume(match):
        nonlocal supported
        fields = dict(
            part.split(b"=", 1) for part in match[1].split(b",") if b"=" in part
        )
        if fields.get(b"i") != str(PROBE_ID).encode():
            return match[0]
        supported = match[2] == b"OK"
        return b""

    pending = REPLY.sub(consume, data)
    return supported, pending


def probe_terminal(stdin, stdout, timeout=0.35):
    if not stdin.isatty() or not stdout.isatty() or os.getenv("TERM") == "dumb":
        return False, ""
    # Query does not display a pixel or leave a placement behind.
    stdout.write(f"\x1b_Gi={PROBE_ID},a=q,t=d,f=24,s=1,v=1;AAAA\x1b\\")
    stdout.flush()
    data = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [stdin], [], [], max(0, deadline - time.monotonic())
        )
        if not ready:
            break
        chunk = os.read(stdin.fileno(), 4096)
        if not chunk:
            break
        data += chunk
        if any(f"i={PROBE_ID}".encode() in m[1] for m in REPLY.finditer(data)):
            break
    ok, pending = parse_probe(data)
    return ok, pending.decode(errors="ignore")


def dimensions(cols, rows):
    """Six real pixels per cell, aspect preserved, bounded GPU/transport cost."""
    width, height = max(16, cols * 6), max(16, (rows - 1) * 12)
    scale = min(1.0, 960 / width, 720 / height)
    return max(16, round(width * scale)), max(16, round(height * scale))


def rgb_png(width, height, rgb):
    if len(rgb) != width * height * 3:
        raise ValueError("invalid RGB frame length")

    def chunk(kind, payload):
        data = kind + payload
        return (
            struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))
        )

    stride = width * 3
    raw = b"".join(b"\0" + rgb[i : i + stride] for i in range(0, len(rgb), stride))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


def frame_payload(scene, width, height, dither=True):
    yaw = scene.yaw + scene.yaw_kick
    pitch = max(0.10, min(0.98, scene.pitch + scene.pitch_kick))
    u = [
        width,
        height,
        scene.t,
        scene.level,
        yaw,
        pitch,
        getattr(scene, "roll", 0.0),
        *scene.tilt_in,
        scene.heat,
        scene.ambient,
        scene.wind,
    ]
    for key in ("cup", "liquid", "crema", "saucer", "steam"):
        # Theme RGB is sRGB; all lighting and compositing happen in linear light.
        u.extend((c / 255.0) ** 2.2 for c in scene.theme[key])
    dots = []
    for p in scene.flat[:420]:
        alpha = max(0.0, 1 - p[3] / max(1.0, p[4]))
        dots.extend(
            (
                p[0] / scene.W,
                p[1] / scene.H,
                max(1.2, width / scene.W * 0.27),
                alpha * 0.65,
            )
        )
    u.extend(
        (
            1.0 if scene.theme_key == "diner" else 0.0,
            1.0 if scene.target_level - scene.level > 0.04 else 0.0,
            len(dots) // 4,
            1.0 if dither else 0.0,
            {"rain": 1.0, "snow": 2.0, "sun": 3.0}.get(scene.weather, 0.0),
        )
    )
    real = getattr(scene, "real_fluid", False)
    u.extend((float(real), 0.9, float(getattr(scene, "cinematic", True)), 0.0))
    gx, gy, gz = getattr(scene, "gravity", (0.0, -1.0, 0.0))
    ax, ay, az = getattr(scene, "fluid_accel", (0.0, 0.0, 0.0))
    c, sn = math.cos(yaw), math.sin(yaw)
    return {
        "u": u,
        "water": [v for row in scene.water.h for v in row],
        "dots": dots,
        "fluid": {
            "gravity": (gx * c + gz * sn, gy, gz * c - gx * sn),
            "accel": (ax * c + az * sn, ay, az * c - ax * sn),
            "stir": float(0.0 <= scene.t - scene.stir_t < (2.8 if real else 1.3)),
            "events": list(getattr(scene, "fluid_events", []))[:32],
            "dt": 0.0,
            "cup_motion": getattr(scene, "cup_motion", True),
        },
    }


class MetalRenderer:
    def __init__(self, root=None):
        root = Path(root or __file__).resolve().parent
        choices = (
            (root / "caf-metal", root / "caf.metal"),
            (root / ".build/caf-metal", root / "native/caf.metal"),
        )
        executable, shader = next(
            ((a, b) for a, b in choices if a.is_file() and b.is_file()), (None, None)
        )
        if executable is None:
            raise OSError("Metal helper not built; run ./install.sh")
        self.proc = subprocess.Popen(
            [str(executable), str(shader)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.last_time = None
        self.stats = {}
        try:
            if self._read(12, 30) != b"CAF_METAL_5\n":
                raise OSError("invalid Metal handshake")
        except BaseException:
            self.close()
            raise

    def _read(self, size, timeout=5):
        result = bytearray()
        deadline = time.monotonic() + timeout
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.proc.stdout], [], [], remaining)[0]
            ):
                raise OSError("Metal renderer timed out")
            data = os.read(self.proc.stdout.fileno(), size - len(result))
            if not data:
                raise OSError("Metal renderer unavailable or stopped")
            result.extend(data)
        return bytes(result)

    def render(self, scene, width, height, dither=True, *, dt=None, draw=True):
        payload = frame_payload(scene, width, height, dither)
        delta = (scene.t - self.last_time) if self.last_time is not None else 0.0
        payload["fluid"]["dt"] = max(0.0, min(0.1, delta if dt is None else dt))
        payload["render"] = draw
        sent_events = len(payload["fluid"]["events"])
        data = (
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        # Blocking writes are bounded too: a stuck helper must not hang caf.
        deadline = time.monotonic() + 5
        fd = self.proc.stdin.fileno()
        os.set_blocking(fd, False)
        try:
            offset = 0
            while offset < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                    raise OSError("Metal input timed out")
                try:
                    offset += os.write(fd, data[offset : offset + 4096])
                except BlockingIOError:
                    continue
        finally:
            os.set_blocking(fd, True)
        (meta_size,) = struct.unpack("<I", self._read(4, 15))
        if not 0 < meta_size <= 65536:
            raise OSError("invalid Metal metadata size")
        stats = json.loads(self._read(meta_size))
        if not isinstance(stats, dict) or stats.get("solver") not in ("legacy", "apic"):
            raise OSError("invalid Metal diagnostics")
        (size,) = struct.unpack("<I", self._read(4))
        if size != (width * height * 3 if draw else 0):
            raise OSError("invalid Metal frame size")
        rgb = self._read(size)
        self.last_time = scene.t
        self.stats = stats
        if stats["solver"] == "apic":
            fraction = stats.get("cup_fraction")
            if (
                not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or fraction < 0
                or stats.get("invalid", 0)
                or stats.get("body_invalid", 0)
            ):
                raise OSError("invalid GPU fluid state; use CPU fallback")
            scene.fluid_stats = stats
            if hasattr(scene, "fluid_feedback"):
                scene.fluid_feedback(stats)
            scene.level = max(0.0, min(1.0, stats["cup_fraction"]))
            if getattr(scene, "fluid_events", None):
                del scene.fluid_events[:sent_events]
        return rgb

    def advance(self, scene, dt=1 / 30):
        self.render(scene, 16, 16, dt=dt, draw=False)
        return self.stats

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for pipe in (proc.stdin, proc.stdout):
            if pipe:
                pipe.close()
        self.proc = None
