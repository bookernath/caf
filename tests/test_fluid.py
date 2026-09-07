"""APIC contracts and actual-GPU acceptance; no sensors or power changes."""

import math
import os
import unittest

from test_caf import ROOT, graphics, lean, level, scene


class FluidContracts(unittest.TestCase):
    def test_live_gravity_both_axes_and_zero(self):
        for axis in (0, 1):
            s = scene()
            level(s)
            lean(s, axis, angle=22, seconds=4)
            self.assertAlmostEqual(
                s.gravity[(0, 2)[axis]], math.sin(math.radians(22)), places=3
            )
            self.assertAlmostEqual(s.gravity[1], -math.cos(math.radians(22)), places=3)
            self.assertAlmostEqual(sum(v * v for v in s.gravity), 1, places=5)
            s.imu_level()
            self.assertEqual(s.gravity, (0.0, -1.0, 0.0))

    def test_controls_and_no_scalar_healing(self):
        s = scene()
        s.real_fluid = True
        s.level = 0.5
        s.sip()
        s.refill()
        s.nudge()
        self.assertEqual([e["kind"] for e in s.fluid_events], ["sip", "refill", "poke"])
        s.step(1 / 30)
        self.assertEqual(s.level, 0.5)
        s.reset_fluid()
        self.assertEqual(s.fluid_events, [{"kind": "reset"}])
        s.level = 0.2
        s.sip()
        self.assertEqual(s.fluid_events[-1]["kind"], "refill")

    def test_screen_gravity_rotates_with_camera(self):
        s = scene()
        s.gravity = (1, 0, 0)
        s.yaw = math.pi / 2
        g = graphics.frame_payload(s, 16, 16)["fluid"]["gravity"]
        self.assertAlmostEqual(g[0], 0)
        self.assertAlmostEqual(g[2], -1)


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires accessible Metal device"
)
class FluidGPU(unittest.TestCase):
    def setUp(self):
        self.gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")
        self.s = scene()
        self.s.real_fluid = True
        self.initial = self.gpu.advance(self.s, 0)["initial"]

    def tearDown(self):
        self.gpu.close()

    def check_mass(self, st):
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
                    "invalid",
                )
            ),
            st["initial"] + st["emitted"],
        )
        self.assertEqual(st["allocated"], st["initial"] + st["emitted"])
        self.assertEqual(st["invalid"], 0)
        self.assertLess(st["max_penetration"], 0.005)
        self.assertLessEqual(st["allocated"], st["capacity"])

    def step(self, n=1, dt=1 / 30):
        for _ in range(n):
            st = self.gpu.advance(self.s, dt)
            self.check_mass(st)
        return st

    def test_upright_pause_resize_and_theme_preserve_volume(self):
        st = self.step(60)
        self.assertEqual(st["in_cup"], self.initial)
        self.assertLess(st["mean_speed2"], 0.005)
        t = st["simulated_time"]
        self.gpu.render(self.s, 200, 140, dt=0)
        self.s.cycle_theme()
        self.gpu.render(self.s, 320, 220, dt=0)
        self.assertEqual(self.gpu.stats["in_cup"], self.initial)
        self.assertEqual(self.gpu.stats["simulated_time"], t)
        self.assertAlmostEqual(self.step(dt=5)["simulated_time"] - t, 0.1, places=5)

    def test_spill_both_axes_and_settle_without_reappearing(self):
        for axis in (0, 2):
            self.s.reset_fluid()
            self.s.gravity = (0, -1, 0)
            self.step(60)
            peak = 0
            for frame in range(210):
                tilt = (
                    22 * min(1, (frame + 1) / 30) * max(0, min(1, (120 - frame) / 30))
                )
                g = [0, -math.cos(math.radians(tilt)), 0]
                g[axis] = math.sin(math.radians(tilt))
                self.s.gravity = g
                st = self.step()
                peak = max(peak, st["mean_speed2"])
            self.assertLess(st["in_cup"], self.initial * 0.96)
            self.assertGreater(st["on_table"] + st["on_saucer"], 200)
            self.assertLess(st["mean_speed2"], peak * 0.02)
            self.assertLess(st["divergence_after"], st["divergence_before"] * 0.15)
            final = st["in_cup"]
            self.assertLessEqual(self.step(30)["in_cup"], final + 10)

    def test_sip_refill_reset_and_pomodoro_sources(self):
        self.s.sip()
        st = self.step(dt=0)
        self.assertGreater(st["sipped"], self.initial * 0.19)
        self.assertEqual(st["emitted"], 0)
        self.assertEqual(self.s.fluid_events, [])
        self.s.refill()
        st = self.step(90)
        self.assertGreater(st["emitted"], 2000)
        self.assertGreater(st["in_cup"], self.initial * 0.88)
        self.s.reset_fluid()
        st = self.step(dt=0)
        self.assertEqual((st["emitted"], st["sipped"], st["off_scene"]), (0, 0, 0))
        self.assertEqual(st["in_cup"], self.initial)
        self.s.fluid_events = [{"kind": "target", "level": 0.4, "fill": False}]
        st = self.step(dt=0)
        self.assertAlmostEqual(st["cup_fraction"], 0.4, places=3)
        # Work must not heal missing volume; break sources are accounted.
        self.s.fluid_events = [{"kind": "target", "level": 0.8, "fill": False}]
        self.assertEqual(self.step(5)["emitted"], 0)
        for _ in range(100):
            self.s.fluid_events = [{"kind": "target", "level": 0.8, "fill": True}]
            st = self.step()
        self.assertGreater(st["cup_fraction"], 0.72)
        self.assertLess(st["emitted"], self.initial * 0.65)

    def test_allocation_capacity_and_reset_recovery(self):
        for _ in range(6):
            self.s.fluid_events = [{"kind": "sip"}] * 5
            self.step(dt=0)
            self.s.refill()
            st = self.step(150)
        self.assertTrue(st["capacity_limited"])
        self.s.reset_fluid()
        st = self.step(dt=0)
        self.assertFalse(st["capacity_limited"])
        self.assertEqual(st["allocated"], self.initial)

    def test_poke_and_stir_move_real_particles(self):
        self.step(30)
        self.s.nudge()
        st = self.step()
        self.assertGreater(st["mean_speed2"], 0.01)
        self.s.reset_fluid()
        self.step(dt=0)
        self.s.stir()
        st = self.step(15)
        self.assertGreater(st["mean_speed2"], 0.005)


if __name__ == "__main__":
    unittest.main()
