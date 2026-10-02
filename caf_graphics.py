"""Optional Metal renderer, Kitty capability probe and frame transport.

No third-party Python dependencies. Importing this module starts no processes
or threads.
"""

import base64
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import threading
import time
from typing import NamedTuple
import zlib

try:
    import _posixshmem  # CPython's POSIX shm binding (multiprocessing.shared_memory)
except ImportError:  # pragma: no cover - non-POSIX builds
    _posixshmem = None

PROBE_ID = 73191
SHM_PROBE_ID = 73192
REPLY = re.compile(rb"\x1b_G([^;]*);([^\x1b]*)\x1b\\")
CELL_REPLY = re.compile(rb"\x1b\[([46]);(\d+);(\d+)t")
DEFAULT_CELL = (6, 12)
# (max width, max height, max pixels). PNG is CPU-bound (deflate + base64
# through the pty, ~10 ms/Mpx even in parallel); shared memory costs only GPU
# time, mostly the resolution-independent fluid solve, so it affords Retina.
PNG_LIMIT = (960, 720, 960 * 720)
SHM_LIMIT = (2560, 1600, 1600 * 1000)


def parse_replies(data, ids=(PROBE_ID,), cells=False):
    """Consume our graphics (and optionally CSI 16t/14t) replies; preserve keys
    and unrelated replies. Returns ({id: ok}, cell_px, window_px, pending)."""
    found, sizes = {}, {}
    wanted = {str(i).encode(): i for i in ids}

    def graphics(match):
        fields = dict(
            part.split(b"=", 1) for part in match[1].split(b",") if b"=" in part
        )
        if fields.get(b"i") not in wanted:
            return match[0]
        found[wanted[fields[b"i"]]] = match[2] == b"OK"
        return b""

    def cell(match):
        sizes[match[1]] = (int(match[3]), int(match[2]))  # replies are h;w
        return b""

    pending = REPLY.sub(graphics, data)
    if cells:
        pending = CELL_REPLY.sub(cell, pending)
    return found, sizes.get(b"6"), sizes.get(b"4"), pending


def parse_probe(data):
    """Consume only our graphics reply; preserve keys and unrelated replies."""
    found, _, _, pending = parse_replies(data)
    return found.get(PROBE_ID, False), pending


def window_pixels(fd):
    """(cols, rows, xpixel, ypixel) from TIOCGWINSZ, or None."""
    try:
        import fcntl
        import termios

        rows, cols, xp, yp = struct.unpack(
            "HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
        )
    except (OSError, ImportError, ValueError):
        return None
    return cols, rows, xp, yp


def plausible_cell(cw, ch):
    return 2 <= cw <= 200 and 4 <= ch <= 400


def cell_pixels(fd):
    """Real cell size in pixels from the kernel window size, or None."""
    size = window_pixels(fd)
    if not size or not all(size):
        return None
    cols, rows, xp, yp = size
    cell = (xp / cols, yp / rows)
    return cell if plausible_cell(*cell) else None


def remote_session(env=None):
    """Over SSH the terminal cannot see this machine's shared memory."""
    env = os.environ if env is None else env
    return any(env.get(k) for k in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))


class Probe(NamedTuple):
    images: bool
    shm: bool
    cell: tuple
    pending: str


def probe_graphics(stdin, stdout, timeout=0.35, shm=True):
    """One round trip: Kitty image support, t=s shared memory, cell pixels.

    Terminals answer in order, so the plain query goes last: once its reply
    arrives, every earlier reply (or silence) is final."""
    if not stdin.isatty() or not stdout.isatty() or os.getenv("TERM") == "dumb":
        return Probe(False, False, DEFAULT_CELL, "")
    cell = cell_pixels(stdout.fileno())
    query, frames = "", None
    if shm and _posixshmem and not remote_session():
        try:
            # Own namespace: a reply that misses the timeout must not let the
            # terminal later consume (unlink) a real frame of the same name.
            frames = SharedFrames(tag="q")
            name = frames.create(3)
            frames.handoff(name)  # a terminal that reads it also unlinks it
            query += (
                f"\x1b_Gi={SHM_PROBE_ID},a=q,t=s,f=24,s=1,v=1,S=3;"
                + base64.standard_b64encode(name.encode()).decode()
                + "\x1b\\"
            )
        except OSError:
            frames = None
    if cell is None:
        query += "\x1b[16t\x1b[14t"
    # Queries display no pixel and leave no placement behind.
    query += f"\x1b_Gi={PROBE_ID},a=q,t=d,f=24,s=1,v=1;AAAA\x1b\\"
    stdout.write(query)
    stdout.flush()
    data = b""
    deadline = time.monotonic() + timeout
    try:
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
    finally:
        if frames:
            frames.close()
    found, cell_reply, window, pending = parse_replies(
        data, (PROBE_ID, SHM_PROBE_ID), cells=cell is None
    )
    if cell is None and cell_reply and plausible_cell(*cell_reply):
        cell = cell_reply
    elif cell is None and window:
        size = window_pixels(stdout.fileno())
        if size and size[0] and size[1]:
            guess = (window[0] / size[0], window[1] / size[1])
            cell = guess if plausible_cell(*guess) else None
    ok = found.get(PROBE_ID, False)
    return Probe(
        ok,
        ok and found.get(SHM_PROBE_ID, False),
        cell or DEFAULT_CELL,
        pending.decode(errors="ignore"),
    )


def probe_terminal(stdin, stdout, timeout=0.35):
    result = probe_graphics(stdin, stdout, timeout, shm=False)
    return result.images, result.pending


def dimensions(cols, rows, cell=DEFAULT_CELL, limit=PNG_LIMIT):
    """Native pixels for the image area (one cell row is the status line),
    aspect preserved, bounded by limit = (max w, max h, max pixels)."""
    width, height = max(16, cols * cell[0]), max(16, (rows - 1) * cell[1])
    scale = min(
        1.0, limit[0] / width, limit[1] / height, math.sqrt(limit[2] / (width * height))
    )
    return max(16, int(width * scale)), max(16, int(height * scale))


_DEFLATE_POOL = None
_DEFLATE_LOCK = threading.Lock()


def _deflate_pool():
    global _DEFLATE_POOL
    with _DEFLATE_LOCK:
        if _DEFLATE_POOL is None:
            from concurrent.futures import ThreadPoolExecutor

            _DEFLATE_POOL = ThreadPoolExecutor(
                max(1, min(4, (os.cpu_count() or 2) - 1)), "caf-deflate"
            )
        return _DEFLATE_POOL


def _deflate_slice(data, last):
    c = zlib.compressobj(1, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush(zlib.Z_FINISH if last else zlib.Z_FULL_FLUSH)


def zlib_level1(raw, segment=256 * 1024):
    """zlib.compress(raw, 1), deflated in parallel slices for large inputs.

    zlib releases the GIL, and full-flushed raw-deflate slices concatenate into
    one valid stream (as pigz does); the reset dictionary costs <1% of size."""
    if len(raw) < 2 * segment:
        return zlib.compress(raw, 1)
    view = memoryview(raw)
    parts = _deflate_pool().map(
        lambda i: _deflate_slice(view[i : i + segment], i + segment >= len(raw)),
        range(0, len(raw), segment),
    )
    return b"\x78\x01" + b"".join(parts) + struct.pack(">I", zlib.adler32(raw))


def rgb_png(width, height, rgb):
    if len(rgb) != width * height * 3:
        raise ValueError("invalid RGB frame length")

    def chunk(kind, payload):
        data = kind + payload
        return (
            struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))
        )

    # Filter 0: per-row Up/Paeth in pure Python costs more than it saves, and
    # level 1 beats the RLE/Huffman-only strategies on both speed and size.
    stride = width * 3
    raw = b"".join(b"\0" + rgb[i : i + stride] for i in range(0, len(rgb), stride))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib_level1(raw))
        + chunk(b"IEND", b"")
    )


def kitty_png(png, cols, rows):
    b64 = base64.standard_b64encode(png).decode()
    parts = ["\x1b[H"]
    step = 4096
    for i in range(0, len(b64), step):
        last = i + step >= len(b64)
        ctl = f"m={0 if last else 1}"
        if i == 0:
            ctl = f"a=T,f=100,i=99,p=1,q=2,c={cols},r={rows}," + ctl
        parts.append(f"\x1b_G{ctl};{b64[i:i + step]}\x1b\\")
    return "".join(parts)


def kitty_shm(name, width, height, cols, rows):
    """Display RGB pixels the terminal reads (then unlinks) from POSIX shm.
    S= is explicit because macOS rounds shm objects up to whole pages."""
    return (
        f"\x1b[H\x1b_Ga=T,f=24,t=s,s={width},v={height},S={width * height * 3},"
        f"i=99,p=1,q=2,c={cols},r={rows};"
        + base64.standard_b64encode(name.encode()).decode()
        + "\x1b\\"
    )


class Frame(NamedTuple):
    """A rendered frame: pixels in rgb, or in the POSIX shm object named shm."""

    width: int
    height: int
    rgb: bytes
    shm: str


def kitty_frame(frame, width, height, cols, rows):
    """Escape sequence for a renderer result: raw RGB bytes (width x height),
    a Frame, or None while a pipelined renderer has nothing ready yet."""
    if frame is None:
        return ""
    if isinstance(frame, Frame):
        if frame.shm:
            return kitty_shm(frame.shm, frame.width, frame.height, cols, rows)
        frame, width, height = frame.rgb, frame.width, frame.height
    return kitty_png(rgb_png(width, height, frame), cols, rows)


class SharedFrames:
    """Per-frame POSIX shm objects for Kitty t=s.

    The terminal unlinks each object after reading it, so every frame needs a
    fresh name. Objects not yet handed off are unlinked on close; handed-off
    ones are reaped `keep` frames later (ENOENT is the normal case), bounding
    memory if a terminal reads without unlinking, or never reads."""

    def __init__(self, keep=16, tag=""):
        if _posixshmem is None:
            raise OSError("POSIX shared memory unavailable")
        self.prefix = f"/caf{os.getpid()}{tag}."  # macOS caps names at 31 bytes
        self.seq = 0
        self.live = set()
        self.sent = deque()
        self.keep = keep
        self.page = os.sysconf("SC_PAGE_SIZE")

    def create(self, size):
        self.seq += 1
        name = f"{self.prefix}{self.seq}"
        fd = _posixshmem.shm_open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        self.live.add(name)
        try:
            # Whole pages: the helper maps it as a no-copy Metal buffer.
            os.ftruncate(fd, -(-size // self.page) * self.page)
        except OSError:
            self.discard(name)
            raise
        finally:
            os.close(fd)
        return name

    def handoff(self, name):
        self.live.discard(name)
        self.sent.append(name)
        while len(self.sent) > self.keep:
            self.unlink(self.sent.popleft())

    def discard(self, name):
        self.live.discard(name)
        self.unlink(name)

    @staticmethod
    def unlink(name):
        try:
            _posixshmem.shm_unlink(name)
        except OSError:
            pass

    def close(self):
        for name in (*self.live, *self.sent):
            self.unlink(name)
        self.live.clear()
        self.sent.clear()


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
        self.transport = "png"  # "shm": pixels go helper -> POSIX shm -> terminal
        self.frames = None
        self._pending = deque()
        self._wait = 0.0
        self._cost = None
        self.pipeline = "off"
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
        """Raw RGB bytes for this scene (the historical contract), or a Frame
        when shared memory carries the pixels.

        With pipelining active this returns the *previous* request's frame (or
        None for the first) so the GPU renders frame N+1 while the caller
        encodes and writes frame N; stats and fluid feedback lag one frame."""
        pipelined = draw and self._pipelined
        if self._pending and not pipelined:
            self._drain()
        ticket = self._send(scene, width, height, dither, dt, draw)
        if not pipelined:
            return self._finish(ticket)
        self._pending.append(ticket)
        if len(self._pending) < 2:
            self._wait = 0.0
            return None
        return self._finish(self._pending.popleft())

    def _send(self, scene, width, height, dither, dt, draw):
        payload = frame_payload(scene, width, height, dither)
        delta = (scene.t - self.last_time) if self.last_time is not None else 0.0
        payload["fluid"]["dt"] = max(0.0, min(0.1, delta if dt is None else dt))
        payload["render"] = draw
        name = None
        if draw and self.transport == "shm":
            try:
                if self.frames is None:
                    self.frames = SharedFrames()
                name = payload["shm"] = self.frames.create(width * height * 3)
            except OSError:
                self.transport = "png"  # out of shm space/descriptors: stay correct
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
        self.last_time = scene.t
        # The helper owns these events now; a pipelined next request must not resend them.
        if payload["u"][32] > 0.5 and getattr(scene, "fluid_events", None):
            del scene.fluid_events[:sent_events]
        return scene, width, height, draw, name

    def _finish(self, ticket):
        scene, width, height, draw, name = ticket
        start = time.perf_counter()
        try:
            (meta_size,) = struct.unpack("<I", self._read(4, 15))
            if not 0 < meta_size <= 65536:
                raise OSError("invalid Metal metadata size")
            stats = json.loads(self._read(meta_size))
            if not isinstance(stats, dict) or stats.get("solver") not in (
                "legacy",
                "apic",
            ):
                raise OSError("invalid Metal diagnostics")
            in_shm = bool(stats.pop("shm", False)) and name is not None
            (size,) = struct.unpack("<I", self._read(4))
            if size != (width * height * 3 if draw and not in_shm else 0):
                raise OSError("invalid Metal frame size")
            rgb = self._read(size)
        except BaseException:
            if name:
                self.frames.discard(name)
            raise
        self._wait = time.perf_counter() - start
        if name and not in_shm:
            self.frames.discard(name)  # helper could not map it; pixels came inline
            self.transport = "png"
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
                if in_shm:
                    self.frames.discard(name)
                raise OSError("invalid GPU fluid state; use CPU fallback")
            scene.fluid_stats = stats
            if hasattr(scene, "fluid_feedback"):
                scene.fluid_feedback(stats)
            scene.level = max(0.0, min(1.0, stats["cup_fraction"]))
        if in_shm:
            # Returned frames are written to the terminal at once; from here
            # the terminal (or the reaper, or close) unlinks the object.
            self.frames.handoff(name)
            return Frame(width, height, None, name)
        if self._pipelined and draw:
            return Frame(width, height, rgb, None)
        return rgb

    def _drain(self):
        while self._pending:
            frame = self._finish(self._pending.popleft())
            if isinstance(frame, Frame) and frame.shm:
                self.frames.unlink(frame.shm)  # never displayed

    def note_frame(self, busy, budget=1 / 30):
        """Adaptive pipelining: pay a frame of latency only when the serial
        GPU + host cost would miss the frame budget. busy is the caller's
        wall time for the frame (render call through terminal write)."""
        if self.pipeline != "auto":
            self._pipelined = self.pipeline == "on"
            return
        cost = max(0.0, busy - self._wait) + self.stats.get("gpu_ms", 0) / 1000
        self._cost = cost if self._cost is None else 0.8 * self._cost + 0.2 * cost
        if self._cost > 0.8 * budget:
            self._pipelined = True
        elif self._cost < 0.55 * budget:
            self._pipelined = False

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, mode):
        if mode not in ("off", "on", "auto"):
            raise ValueError("pipeline must be off, on or auto")
        self._pipeline = mode
        self._pipelined = mode == "on"

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
        self._pending.clear()
        if self.frames:
            self.frames.close()
