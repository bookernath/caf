"""Conservative spray, wet-film transport and aging, using real production kernels."""

import json
import math
import os
import subprocess
import unittest

from test_caf import ROOT, graphics, scene


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires accessible Metal device"
)
class DetailKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = ROOT / ".build/detail-kernels"
        subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-O",
                "-module-cache-path",
                str(ROOT / ".build/module-cache"),
                str(ROOT / "tests/detail_kernels.swift"),
                "-o",
                str(cls.helper),
            ],
            check=True,
            capture_output=True,
        )

    def probe(self, mode):
        result = subprocess.run(
            [str(self.helper), str(ROOT / "native"), mode],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        return json.loads(result.stdout)

    def test_split_preserves_volume_momentum_and_capacity(self):
        st = self.probe("split")
        self.assertEqual(st["slots"], 8)
        self.assertEqual(st["volume"], 1)
        for actual, expected in zip(st["momentum"], (2, 1, -4)):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertLess(st["momentum_error"], 10)
        self.assertEqual(st["capacity_slots"], 119997)
        self.assertEqual(st["unsplit_weight"], 8)

    def test_film_transport_evaporation_and_pigment_are_conservative(self):
        st = self.probe("dry")
        self.assertEqual(st["water"] + st["evaporated"], st["initial"])
        self.assertEqual(st["pigment"], st["initial"])
        self.assertGreater(st["evaporated"], 0)
        self.assertGreater(st["residue"], 0)
        self.assertGreater(st["stained_cells"], 0)
        self.assertGreater(
            st["wet_cells"], 193
        )  # source disk spreads, not a stamped stain

    def test_ripples_decay_without_continuing_impacts(self):
        st = self.probe("ripple")
        self.assertGreater(st["peak"], 1e-6)
        self.assertLess(st["final_wave"], st["peak"] * 0.01)
        self.assertLess(st["final_velocity"], 0.0001)


@unittest.skipUnless(
    os.getenv("CAF_TEST_METAL") == "1", "requires accessible Metal device"
)
class DetailGPU(unittest.TestCase):
    def setUp(self):
        self.gpu = graphics.MetalRenderer(ROOT / "caf_graphics.py")
        self.s = scene()
        self.s.real_fluid = True
        self.s.yaw = math.pi / 2
        self.initial = self.gpu.advance(self.s, 0)["initial"]

    def tearDown(self):
        self.gpu.close()

    def step(self, n=1, dt=1 / 30):
        for _ in range(n):
            self.s.t += dt
            st = self.gpu.advance(self.s, dt)
            self.assertEqual(st["invalid"], 0)
            self.assertEqual(st["body_invalid"], 0)
            self.assertEqual(st["film_balance_error"], 0)
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
            self.assertLessEqual(st["slots_used"], st["slot_capacity"])
            self.assertLess(st["max_penetration"], 0.005)
            self.assertLessEqual(st["max_film_depth"],.03201)
        return st

    def test_rest_does_not_invent_droplets_or_stains(self):
        st = self.step(60)
        self.assertEqual(st["in_cup"], self.initial)
        for key in (
            "split_events",
            "spray_droplets",
            "deposited_volume",
            "film_quanta",
            "evaporated_quanta",
            "stained_cells",
        ):
            self.assertEqual(st[key], 0, key)

    def test_actual_knock_spray_wetness_pause_aging_and_reset(self):
        self.step(30)
        self.s.knock_over()
        peak = 0
        for _ in range(180):
            st = self.step()
            peak = max(peak, st["spray_droplets"])
            if st["spray_droplets"]:
                self.assertLess(st["spray_max_radius"], 0.014)
        self.assertGreater(peak, 100)
        self.assertGreater(st["rim_drips"], 0)
        self.assertGreater(st["cup_ripples"], 0)
        self.assertGreater(st["deposited_volume"], 100)
        self.assertGreater(st["wet_cells"], 500)
        self.assertGreater(st["surface_impacts"], 0)
        self.assertLess(st["split_momentum_error"], 0.00001)
        snapshot = dict(st)
        for size in ((200, 140), (240, 160)):
            self.gpu.render(self.s, *size, dt=0)
            for key in (
                "film_quanta",
                "evaporated_quanta",
                "deposited_volume",
                "slots_used",
                "split_events",
                "simulated_time",
            ):
                self.assertEqual(self.gpu.stats[key], snapshot[key], key)
        st = self.step(200, dt=0.1)
        self.assertGreater(st["evaporated_quanta"], snapshot["evaporated_quanta"])
        self.assertGreater(st["coffee_residue"], snapshot["coffee_residue"])
        self.assertGreater(st["edge_residue"], 0)
        self.s.reset_fluid()
        st = self.gpu.advance(self.s, 0)
        for key in (
            "split_events",
            "spray_droplets",
            "deposited_volume",
            "film_quanta",
            "evaporated_quanta",
            "coffee_residue",
            "stained_cells",
        ):
            self.assertEqual(st[key], 0, key)
        self.assertEqual(st["slots_used"], self.initial)
        self.assertEqual(st["in_cup"], self.initial)

    def test_cream_appears_in_residue_without_creating_material(self):
        self.step(30)
        self.s.add_cream()
        self.step(75)
        expected = self.gpu.stats["cream_emitted"]
        self.s.knock_over()
        st = self.step(250)
        self.assertGreater(st["cream_residue"], 0)
        self.assertLess(abs(st["cream_mass"] - expected), expected * 0.015)


if __name__ == "__main__":
    unittest.main()
