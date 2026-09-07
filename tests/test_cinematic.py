"""Cinematic/cream/rigid-cup acceptance. Never starts real sensors or audio."""

import io
import json
import math
import os
import pty
import select
import subprocess
import time
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from test_caf import CAF, ROOT, graphics, scene


class CinematicContracts(unittest.TestCase):
    def test_new_controls_and_locked_cup(self):
        s = scene()
        s.real_fluid = True
        s.add_cream()
        s.knock_over()
        self.assertEqual(s.fluid_events, [{"kind": "cream"}, {"kind": "knock"}])
        s.cup_motion = False
        s.knock_over()
        self.assertEqual(len(s.fluid_events), 2)
        s.cinematic = False
        p = graphics.frame_payload(s, 16, 16)
        self.assertFalse(p["fluid"]["cup_motion"])
        self.assertEqual(p["u"][34], 0)
        s.reset_fluid()
        self.assertEqual(s.fluid_events, [{"kind": "reset"}])

    def test_input_batches_bound_the_native_contract(self):
        s = scene()
        s.fluid_events = [{"kind": "cream"}] * 80
        p = graphics.frame_payload(s, 16, 16)
        self.assertEqual(len(p["fluid"]["events"]), 32)
        self.assertEqual(len(s.fluid_events), 80)  # consume only after successful reply

    def test_physics_audio_is_event_driven_and_rate_limited(self):
        s = scene()
        heard = []
        s.snd = lambda name, volume: heard.append((name, volume))
        s.fluid_feedback({"simulated_time": 1, "cup_impact": 0, "splash_impact": 0})
        self.assertEqual(heard, [])
        s.fluid_feedback({"simulated_time": 1, "cup_impact": 2, "splash_impact": 100})
        self.assertEqual([x[0] for x in heard], ["ceramic", "splash"])
        s.fluid_feedback(
            {"simulated_time": 1.01, "cup_impact": 2, "splash_impact": 100}
        )
        self.assertEqual(len(heard), 2)
        self.assertTrue(all(0 < v <= 0.65 for _, v in heard))
        s.fluid_feedback({"simulated_time": 0, "cup_impact": 2})
        self.assertEqual(len(heard), 3)  # reset clears stale cooldowns

    def test_foley_samples_and_bounded_process_cleanup(self):
        SoundKit = CAF["SoundKit"]
        with tempfile.TemporaryDirectory() as d:
            kit = SoundKit.__new__(SoundKit)
            kit.dir = d
            kit.ok = True
            kit._last = {}
            kit._voices = []
            kit._generate()
            for name in ("ceramic", "splash"):
                with wave.open(str(Path(d) / (name + ".wav"))) as wav:
                    self.assertEqual(wav.getnchannels(), 1)
                    self.assertEqual(wav.getsampwidth(), 2)
                    self.assertGreater(wav.getnframes(), 1000)
                    self.assertTrue(any(wav.readframes(1000)))
            voices = [Mock() for _ in range(3)]
            for p in voices:
                p.poll.return_value = None
            with patch.dict(
                kit.play.__globals__, subprocess=Mock(Popen=Mock(side_effect=voices))
            ):
                for name in ("ceramic", "splash", "clink", "drip"):
                    kit.play(name)
            self.assertEqual(len(kit._voices), 3)
            kit.close()
            for p in voices:
                p.terminate.assert_called_once()
                p.wait.assert_called_once()
            self.assertEqual(kit._voices, [])

    def test_cli_cup_and_studio_reach_renderer(self):
        seen = []

        class GPU:
            def render(self, s, w, h, dither):
                seen.append((s.cup_motion, s.cinematic, s.real_fluid))
                return bytes(w * h * 3)

            def close(self):
                pass

        with (
            patch.dict(CAF["main"].__globals__, MetalRenderer=GPU),
            patch.object(
                sys,
                "argv",
                [
                    "caf",
                    "--demo",
                    "1",
                    "--size",
                    "30x12",
                    "--renderer",
                    "metal",
                    "--cup",
                    "locked",
                    "--studio",
                ],
            ),
            patch.object(sys, "stdout", io.StringIO()),
        ):
            CAF["main"]()
        self.assertEqual(seen, [(False, False, True)])


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires accessible Metal device"
)
class CinematicGPU(unittest.TestCase):
    def setUp(self):
        self.gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")
        self.s = scene()
        self.s.real_fluid = True
        self.initial = self.gpu.advance(self.s, 0)["initial"]

    def tearDown(self):
        self.gpu.close()

    def step(self, frames=1, dt=1 / 30):
        for _ in range(frames):
            self.s.t += dt
            st = self.gpu.advance(self.s, dt)
            self.assertEqual(st["invalid"], 0)
            self.assertEqual(st["body_invalid"], 0)
            self.assertLess(st["max_penetration"], 0.005)
            self.assertLess(st["cup_penetration"], 0.005)
            self.assertEqual(
                sum(
                    st[k]
                    for k in (
                        "in_cup",
                        "on_saucer",
                        "on_table",
                        "airborne",
                        "off_scene",
                        "sipped",
                    )
                ),
                st["allocated"],
            )
            self.assertEqual(st["allocated"], st["initial"] + st["emitted"])
            self.assertTrue(
                all(math.isfinite(x) for x in st["cup_rotation"] + st["cup_position"])
            )
            self.assertAlmostEqual(sum(x * x for x in st["cup_rotation"]), 1, places=4)
        return st

    def test_cup_rest_and_volume_do_not_drift(self):
        st = self.step(300)
        self.assertEqual(st["in_cup"], self.initial)
        self.assertLess(st["cup_tilt"], 0.3)
        self.assertLess(math.hypot(st["cup_position"][0], st["cup_position"][2]), 0.015)
        self.assertLess(st["mean_speed2"], 0.005)
        self.assertGreater(st["grid_volume"], self.initial * 0.0425**3 * 0.85)
        self.assertLess(st["max_grid_density"], 10)

    def test_cream_mass_is_advected_and_mixed_not_painted(self):
        self.step(30)
        self.s.cinematic = False
        before = self.gpu.render(self.s, 320, 220, dt=0)
        self.s.add_cream()
        paused = self.step(dt=0)
        self.assertEqual(paused["cream_emitted"], 0)
        self.assertGreater(paused["cream_remaining"], 0)
        st = self.step(90)
        self.assertEqual(st["cream_remaining"], 0)
        self.assertGreater(st["cream_emitted"], 800)
        self.assertLess(
            abs(st["cream_mass"] - st["cream_emitted"]), st["cream_emitted"] * 0.01
        )
        variance = st["cream_variance"]
        self.s.stir()
        st = self.step(180)
        self.assertGreater(st["cream_fraction"], 0.03)
        self.assertLess(st["cream_variance"], variance * 0.8)
        self.assertGreater(st["grid_volume"], st["allocated"] * 0.0425**3 * 0.8)
        self.assertLess(st["max_grid_density"], 16)
        after = self.gpu.render(self.s, 320, 220, dt=0)
        self.assertNotEqual(before, after)
        self.s.reset_fluid()
        st = self.step(dt=0)
        self.assertEqual(
            (st["cream_emitted"], st["cream_remaining"], st["cream_mass"]), (0, 0, 0)
        )

    def test_knock_tips_and_dumps_cup_in_two_directions(self):
        for yaw in (math.pi, math.pi / 2):
            self.s.yaw = yaw
            self.s.reset_fluid()
            self.step(45)
            self.s.knock_over()
            peak = 0
            impact = 0
            for _ in range(300):
                st = self.step()
                peak = max(peak, st["cup_tilt"])
                impact = max(impact, st["cup_impact"])
            self.assertGreater(peak, 55)
            self.assertGreater(impact, 0.5)
            self.assertLess(st["in_cup"], self.initial * 0.4)
            self.assertGreater(
                st["on_table"] + st["on_saucer"] + st["off_scene"], self.initial * 0.5
            )
            self.assertLess(sum(x * x for x in st["cup_velocity"]), 0.01)
            self.assertLess(sum(x * x for x in st["cup_angular"]), 0.01)
            pose = st["cup_rotation"]
            self.gpu.render(self.s, 200, 140, dt=0)
            self.assertEqual(self.gpu.stats["cup_rotation"], pose)

    def test_handle_can_arrest_a_knock_and_zero_g_remains_finite(self):
        self.s.yaw = 0.0  # toward the handle: it can catch the saucer
        self.step(30)
        self.s.knock_over()
        peak = 0.0
        for _ in range(210):
            st = self.step()
            peak = max(peak, st["cup_tilt"])
        self.assertGreater(peak, 10.0)
        self.assertLess(peak, 55.0)
        self.assertLess(sum(x * x for x in st["cup_velocity"]), 0.01)
        self.s.reset_fluid()
        self.step(dt=0)
        self.s.fluid_accel = (0.0, -20.0, 0.0)  # cancels gravity exactly
        self.step(15)
        self.s.fluid_accel = (0.0, 0.0, 0.0)
        self.step(15)

    def test_locked_cup_and_light_toggle_do_not_reset_simulation(self):
        self.s.cup_motion = False
        self.s.fluid_events = [{"kind": "knock"}]
        st = self.step(60)
        self.assertLess(st["cup_tilt"], 0.001)
        self.assertAlmostEqual(st["cup_position"][1], 0.52, places=5)
        a = self.gpu.render(self.s, 200, 140, dt=0)
        snapshot = self.gpu.stats["simulated_time"]
        self.s.cinematic = False
        b = self.gpu.render(self.s, 200, 140, dt=0)
        self.assertNotEqual(a, b)
        self.assertEqual(self.gpu.stats["in_cup"], self.initial)
        self.assertEqual(self.gpu.stats["simulated_time"], snapshot)
        self.s.cinematic = True
        self.s.t += 5
        c = self.gpu.render(self.s, 200, 140, dt=0)
        self.assertNotEqual(a, c)
        self.assertEqual(self.gpu.stats["simulated_time"], snapshot)

    def test_repeated_stirring_preserves_occupied_volume(self):
        self.s.cup_motion = False
        self.step(45)
        for _ in range(3):
            self.s.stir()
            st = self.step(180)
            alive = st["allocated"] - st["off_scene"] - st["sipped"]
            self.assertGreater(st["grid_volume"], alive * 0.0425**3 * 0.8)
            self.assertGreater(st["mean_cup_height"], 0.44)
            self.assertLess(st["max_grid_density"], 16)


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires accessible Metal device"
)
class CinematicTerminal(unittest.TestCase):
    def test_real_gpu_keyboard_cream_knock_light_reset_and_quit(self):
        script = """import json,runpy,sys
m=runpy.run_path(sys.argv[1],run_name='caf_pty')
g=m['main'].__globals__
g['imu_start_bridge']=lambda:None
g['imu_read_sample']=lambda:None
base=g['MetalRenderer']
class GPU(base):
    def render(self,s,w,h,dither=True,**kw):
        events=[e['kind'] for e in s.fluid_events]
        rgb=super().render(s,w,h,dither,**kw)
        with open(log,'a') as f:
            f.write(json.dumps(dict(self.stats,events=events,cinematic=s.cinematic))+'\\n')
        return rgb
log=sys.argv[2]
g['MetalRenderer']=GPU
sys.argv=['caf','--renderer','metal','--no-caffeinate','--no-lid-guard','--size','60x24']
raise SystemExit(g['main']())
"""
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "frames.jsonl"
            master, slave = pty.openpty()
            proc = subprocess.Popen(
                [sys.executable, "-c", script, str(ROOT / "caf-art"), str(log)],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=ROOT,
                env=dict(os.environ, TERM="xterm-256color"),
            )
            os.close(slave)
            output = b""
            phase = 0
            replied = False
            records = []
            deadline = time.monotonic() + 15
            try:
                while time.monotonic() < deadline and proc.poll() is None:
                    if select.select([master], [], [], 0.05)[0]:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError:
                            break
                        output = (output + chunk)[-500000:]
                    if b"a=q" in output and not replied:
                        os.write(master, b"\x1b_Gi=73191;OK\x1b\\")
                        replied = True
                    if log.exists():
                        records = []
                        for line in log.read_text().splitlines(keepends=True):
                            if line.endswith("\n"):
                                records.append(json.loads(line))
                    if not records:
                        continue
                    st = records[-1]
                    if phase == 0:
                        os.write(master, b"m ")
                        phase = 1
                    elif phase == 1 and st.get("cream_emitted", 0) > 30:
                        os.write(master, b"k")
                        phase = 2
                    elif phase == 2 and st.get("cup_tilt", 0) > 5:
                        os.write(master, b"l")
                        phase = 3
                    elif phase == 3 and not st.get("cinematic", True):
                        os.write(master, b"r")
                        phase = 4
                    elif (
                        phase == 4
                        and st.get("cream_emitted") == 0
                        and st.get("cup_tilt", 99) < 1
                    ):
                        os.write(master, b"q")
                        phase = 5
                proc.wait(timeout=2)
                while select.select([master], [], [], 0.05)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output = (output + chunk)[-500000:]
                self.assertEqual(proc.returncode, 0, output[-2000:])
                self.assertEqual(phase, 5, output[-2000:])
                self.assertTrue(replied)
                self.assertIn(b"\x1b[?1049l", output)
                self.assertIn(b"a=d,d=I,i=99", output)
                seen = {event for r in records for event in r["events"]}
                self.assertTrue({"cream", "knock", "reset"} <= seen)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                os.close(master)


if __name__ == "__main__":
    unittest.main()
