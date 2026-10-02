"""Watery optics and focus-aware idle regressions; no real sensors/power/audio."""

import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tempfile
import time
import unittest

from test_caf import ROOT, motion, scene, graphics


class FocusTests(unittest.TestCase):
    def test_sensor_pause_discards_samples_and_stops_polling(self):
        calls = []

        def reader():
            calls.append(time.monotonic())
            return (time.monotonic(), (0, 0, 1), (0, 0, 0), -1)

        sampler = motion.SensorSamples(reader)
        try:
            time.sleep(0.04)
            sampler.set_paused(True)
            time.sleep(0.025)  # allow one already-started read to finish
            n = len(calls)
            time.sleep(0.08)
            self.assertEqual(len(calls), n)
            self.assertEqual(sampler.drain(), [])
            sampler.set_paused(False)
            time.sleep(0.04)
            self.assertGreater(len(calls), n)
            self.assertTrue(sampler.drain())
            sampler.set_paused(True)
        finally:
            sampler.close()
        self.assertFalse(sampler.thread.is_alive())

    def test_resume_reacquires_pose_without_recalibrating_level(self):
        s = scene()
        s._imu_up = [0.0, 0.0, 1.0]
        s._imu_rest = [0.0, 0.0, 1.0]
        s._imu_prev_a = [0.0, 0.0, 1.0]
        original = list(s._imu_rest)
        s.resume_imu((0.6, 0.0, 0.8))
        self.assertEqual(s._imu_rest, original)
        self.assertEqual(s._imu_up, [0.6, 0.0, 0.8])
        self.assertIsNone(s._imu_prev_a)
        self.assertEqual(s.fluid_accel, (0.0, 0.0, 0.0))

    def run_focus_session(self, real_gpu=False):
        script = r"""
import json,runpy,sys,time
from pathlib import Path
m=runpy.run_path(sys.argv[1],run_name='focus_test')
g=m['main'].__globals__
log,offset=Path(sys.argv[2]),Path(sys.argv[3])
def record(kind,**kw):
    with log.open('a') as f: f.write(json.dumps(dict(kind=kind,**kw))+'\n')
class Clock:
    def monotonic(self): return time.monotonic()+float(offset.read_text())
    def __getattr__(self,k): return getattr(time,k)
g['time']=Clock()
g['imu_start_bridge']=lambda:None
g['imu_read_sample']=lambda:None
class Sensors:
    def __init__(self,*a): pass
    def drain(self): return []
    def set_paused(self,paused): record('paused',value=paused)
    def close(self): record('sensors_closed')
g['SensorSamples']=Sensors
class Kit:
    ok=True
    def play(self,*args): record('sound',name=args[0])
    def close(self): record('voices_stopped')
g['SoundKit']=Kit
base=g['MetalRenderer'] if sys.argv[4]=='1' else object
class GPU(base):
    def render(self,s,w,h,dither=True,**kw):
        events=list(s.fluid_events)
        if base is object:
            rgb=bytes(w*h*3);s.fluid_events.clear();sim=s.t
        else:
            rgb=super().render(s,w,h,dither,**kw);sim=self.stats['simulated_time']
        record('frame',t=s.t,sim=sim,events=events,film=0 if base is object else self.stats['film_quanta'],evap=0 if base is object else self.stats['evaporated_quanta'])
        return rgb
    def close(self):
        if base is not object: super().close()
        record('gpu_closed')
g['MetalRenderer']=GPU
orig=g['Scene'].write_word
def word(self,value):
    record('word',text=value);return orig(self,value)
g['Scene'].write_word=word
sys.argv=['caf','--renderer','metal','--no-caffeinate','--no-lid-guard',
          '--size','40x16','--sound','--pomodoro','1']
raise SystemExit(g['main']())
"""
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "events.jsonl"
            offset = Path(d) / "offset"
            offset.write_text("0")
            master, slave = pty.openpty()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(ROOT / "caf-art"),
                    str(log),
                    str(offset),
                    str(int(real_gpu)),
                ],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=ROOT,
                # Pinned serial: the pause assertions compare each frame's own
                # stats, which pipelining (auto under GPU load) shifts by one.
                env=dict(os.environ, TERM="xterm-256color", CAF_PIPELINE="off"),
            )
            os.close(slave)
            output = bytearray()
            replied = False

            def records():
                if not log.exists():
                    return []
                return [
                    json.loads(line)
                    for line in log.read_text().splitlines(keepends=True)
                    if line.endswith("\n")
                ]

            def pump(seconds):
                nonlocal replied
                end = time.monotonic() + seconds
                while time.monotonic() < end and proc.poll() is None:
                    if select.select(
                        [master], [], [], min(0.02, max(0, end - time.monotonic()))
                    )[0]:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            break
                    if not replied and b"a=q" in output:
                        os.write(master, b"\x1b_Gi=73191;OK\x1b\\")
                        replied = True

            def until(predicate, timeout=8):
                end = time.monotonic() + timeout
                while time.monotonic() < end and not predicate():
                    pump(0.03)
                self.assertTrue(
                    predicate(), bytes(output[-3000:]).decode(errors="replace")
                )

            try:
                until(lambda: len([r for r in records() if r["kind"] == "frame"]) >= 4)
                self.assertIn(b"\x1b[?1004h", output)
                if real_gpu:
                    os.write(master, b"k")
                    until(lambda: any(r["kind"] == "frame" and r["film"] > 0 for r in records()))
                # Split the CSI across reads; partial focus bytes must not be keys.
                os.write(master, b"\x1b")
                pump(0.06)
                os.write(master, b"[")
                pump(0.06)
                os.write(master, b"O")
                until(
                    lambda: any(r["kind"] == "paused" and r["value"] for r in records())
                )
                pump(0.1)
                frozen = [r for r in records() if r["kind"] == "frame"]
                sounds_before_pause = [r for r in records() if r["kind"] == "sound"]
                self.assertTrue(frozen)
                output.clear()
                os.write(master, b"m")  # accidental background input is not applied
                update = offset.with_suffix(".new")
                update.write_text("61")
                update.replace(offset)  # atomic clock jump
                pump(1.3)
                self.assertEqual([r for r in records() if r["kind"] == "frame"], frozen)
                self.assertNotIn(b"\x1b[13t", output)
                self.assertNotIn(b"a=T", output)
                self.assertTrue(
                    any(r["kind"] == "word" and r["text"] == "BREAK" for r in records())
                )
                self.assertEqual([r for r in records() if r["kind"] == "sound"], sounds_before_pause)
                self.assertFalse(any(r["kind"] == "gpu_closed" for r in records()))
                os.write(master, b"\x1b[I")
                until(
                    lambda: len([r for r in records() if r["kind"] == "frame"])
                    > len(frozen)
                )
                resumed = [r for r in records() if r["kind"] == "frame"][len(frozen)]
                self.assertLessEqual(resumed["t"] - frozen[-1]["t"], 1 / 30 + 1e-6)
                self.assertLessEqual(resumed["sim"] - frozen[-1]["sim"], 1 / 30 + 1e-5)
                self.assertNotIn("cream", [e["kind"] for e in resumed["events"]])
                self.assertEqual(resumed["film"], frozen[-1]["film"])
                self.assertEqual(resumed["evap"], frozen[-1]["evap"])
                targets = [e for e in resumed["events"] if e["kind"] == "target"]
                self.assertEqual(len(targets), 1)
                self.assertTrue(targets[0]["fill"])  # timer advanced to break
                os.write(master, b"\x1b[O")
                pump(0.1)
                os.write(master, b"q")  # quitting still works while blurred
                until(lambda: proc.poll() is not None)
                self.assertEqual(proc.returncode, 0)
                # Read any final bytes left after the process exits.
                while select.select([master], [], [], 0)[0]:
                    try:
                        chunk = os.read(master, 65536)
                        if not chunk:
                            break
                        output.extend(chunk)
                    except OSError:
                        break
                self.assertIn(b"\x1b[?1004l", output)
                self.assertIn(b"\x1b[?1049l", output)
                self.assertTrue(any(r["kind"] == "sensors_closed" for r in records()))
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                os.close(master)

    def test_focus_pauses_without_timer_or_resume_backlog(self):
        self.run_focus_session()

    @unittest.skipUnless(os.getenv("CAF_TEST_METAL") == "1", "requires actual GPU")
    def test_real_gpu_focus_pause_resume(self):
        self.run_focus_session(real_gpu=True)


@unittest.skipUnless(os.getenv("CAF_TEST_METAL") == "1", "requires actual GPU")
class WateryGPU(unittest.TestCase):
    def test_falling_coffee_uses_mug_scale_gravity(self):
        gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")
        s = scene()
        s.real_fluid = True
        s.cup_motion = False
        try:
            gpu.advance(s, 0)
            s.fluid_events = [
                {"kind": "sip"}
            ] * 6  # integer sip rounding can leave 1–4 particles
            gpu.advance(s, 0)
            self.assertEqual(gpu.stats["in_cup"], 0)
            s.refill()
            st = gpu.advance(s, 0.01)
            self.assertGreater(st["emitted"], 0)
            self.assertEqual(st["invalid"], 0)
            # Fresh coffee leaves at 6.64 model units/s and accelerates by
            # 98.1*.01, before reaching the rim. No visual/color test required.
            self.assertAlmostEqual(st["max_speed"], 6.64 + 98.1 * 0.01, delta=0.20)
        finally:
            gpu.close()


if __name__ == "__main__":
    unittest.main()
