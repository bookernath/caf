"""Directional calibration and timestamped sensor buffering; no side effects on import."""

from collections import deque
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError("invalid sensor vector")
    length = math.sqrt(dot(v, v))
    if length < 1e-5:
        raise ValueError("not enough directional movement")
    return tuple(x / length for x in v)


def projected(v, up):
    return tuple(v[i] - dot(v, up) * up[i] for i in range(3))


def calibrate(level, right_down, far_down):
    """Recover screen axes from deliberate poses, not arbitrary sensor indices."""
    up = unit(level)
    right_down, far_down = unit(right_down), unit(far_down)
    for pose in (right_down, far_down):
        angle = math.acos(max(-1.0, min(1.0, dot(up, pose))))
        if not math.radians(6) <= angle <= math.radians(40):
            raise ValueError("tilt each direction between 6 and 40 degrees")
    right = unit(tuple(-v for v in projected(right_down, up)))
    near = unit(projected(far_down, up))  # far edge down means negative near tilt
    if abs(dot(right, near)) > 0.5:
        raise ValueError("poses are not independent; tilt sideways, then forward/back")
    near = unit(projected(near, right))
    return {"version": 1, "right": right, "near": near, "up": up}


def calibration_path():
    return Path(os.path.expanduser("~/.config/caf/imu.json"))


def load_calibration(path=None):
    try:
        data = json.loads(Path(path or calibration_path()).read_text())
        if not isinstance(data, dict) or data.get("version") != 1:
            return None
        axes = [unit(data[k]) for k in ("right", "near", "up")]
        if any(abs(dot(axes[i], axes[j])) > 0.02 for i in range(3) for j in range(i)):
            return None
        return dict(zip(("right", "near", "up"), axes), version=1)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def save_calibration(data, path=None):
    path = Path(path or calibration_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".imu-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(data, out, indent=2)
            out.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def basis_for_rest(calibration, rest):
    right = unit(projected(calibration["right"], rest))
    near = unit(projected(projected(calibration["near"], rest), right))
    return right, near, tuple(rest)


class SensorSamples:
    """Poll independently of rendering; drain each unique sample exactly once.

    The bridge publishes ~100 Hz. A bounded queue preserves short impulses
    through a slow frame without replaying seconds of motion after a stall.
    Scene mutation stays on the main thread.
    """

    def __init__(self, reader):
        self.reader = reader
        self.queue = deque(maxlen=128)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()
        self.thread = threading.Thread(
            target=self._run, name="caf-sensors", daemon=True
        )
        self.thread.start()

    def _run(self):
        last = None
        while not self.stop.is_set():
            if not self.enabled.wait(0.5) or self.stop.is_set():
                continue
            sample = self.reader()
            if sample and sample[0] != last and abs(time.monotonic() - sample[0]) < 0.6:
                with self.lock:
                    if self.enabled.is_set():
                        self.queue.append(sample)
                last = sample[0]
            self.stop.wait(0.008)

    def drain(self):
        with self.lock:
            samples = list(self.queue)
            self.queue.clear()
        return samples

    def set_paused(self, paused):
        with self.lock:
            self.queue.clear()
            if paused:
                self.enabled.clear()
            else:
                self.enabled.set()

    def close(self):
        self.stop.set()
        self.enabled.set()
        self.thread.join(timeout=1)


def capture_pose(reader, seconds=0.7):
    values = []
    last = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = reader()
        if s and s[0] != last and abs(time.monotonic() - s[0]) < 0.3:
            a = s[1]
            mag = math.sqrt(dot(a, a))
            if 0.85 < mag < 1.15 and math.sqrt(dot(s[2], s[2])) < 8:
                values.append(unit(a))
            last = s[0]
        time.sleep(0.01)
    if len(values) < 8:
        raise ValueError("no stable sensor stream; hold still or run caf --imu-debug")
    average = unit(tuple(sum(v[i] for v in values) / len(values) for i in range(3)))
    if max(
        math.acos(max(-1.0, min(1.0, dot(v, average)))) for v in values
    ) > math.radians(3):
        raise ValueError("pose moved during capture; hold still and try again")
    return average


def calibration_wizard(reader, start_bridge):
    proc = start_bridge()
    try:
        print(
            "caf calibration — hold each pose still while it is sampled. Ctrl+C cancels."
        )
        print(
            "Move the whole laptop gently; do not twist the display or lift it steeply."
        )
        poses = []
        for prompt in (
            "Rest the laptop in your usual level position",
            "Lower the RIGHT edge about 10–20 degrees",
            "Return to level, then lower the FAR edge about 10–20 degrees",
        ):
            input(prompt + "; press Enter and hold still: ")
            poses.append(capture_pose(reader))
        data = calibrate(*poses)
        save_calibration(data)
        print(
            "Saved directional calibration. Start caf level; z re-zeros without losing the axis mapping."
        )
        return 0
    except (ValueError, OSError, EOFError) as exc:
        print(f"Calibration not saved: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCalibration cancelled; previous mapping preserved.")
        return 1
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except Exception:
                proc.kill()
                proc.wait()
