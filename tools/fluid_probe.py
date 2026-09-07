#!/usr/bin/env python3
"""Deterministic APIC acceptance probe; synthetic gravity, no sensors/power changes."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_graphics import MetalRenderer, rgb_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--tilt", type=float, default=40)
    ap.add_argument("--axis", choices=["x", "z"], default="x")
    ap.add_argument("--output", default=str(ROOT / ".build/fluid"))
    ap.add_argument("--render-every", type=int, default=60)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = runpy.run_path(str(ROOT / "caf-art"), run_name="fluid_probe")["Scene"](
        "diner"
    )
    scene.mode3d = True
    scene.real_fluid = True
    scene.layout(90, 31)
    scene.ambient = 0.8
    scene.heat = 0.15
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    records = []
    try:
        for frame in range(args.frames + 1):
            angle = 0.0
            if 60 <= frame < 90:
                angle = args.tilt * (frame - 60) / 30
            elif 90 <= frame < 150:
                angle = args.tilt
            elif 150 <= frame < 180:
                angle = args.tilt * (180 - frame) / 30
            a = math.radians(angle)
            scene.gravity = (
                (math.sin(a), -math.cos(a), 0.0)
                if args.axis == "x"
                else (0.0, -math.cos(a), math.sin(a))
            )
            scene.t = frame / 30
            start = time.perf_counter()
            stats = gpu.advance(scene, 0.0 if frame == 0 else 1 / 30)
            stats["frame"] = frame
            stats["tilt"] = angle
            stats["wall_ms"] = (time.perf_counter() - start) * 1000
            records.append(dict(stats))
            if frame % 15 == 0:
                print(
                    json.dumps(
                        {
                            k: stats[k]
                            for k in (
                                "frame",
                                "in_cup",
                                "on_saucer",
                                "on_table",
                                "airborne",
                                "off_scene",
                                "invalid",
                                "mean_speed2",
                                "max_penetration",
                                "divergence_before",
                                "divergence_after",
                                "gpu_ms",
                            )
                        }
                    ),
                    flush=True,
                )
            if args.render_every and frame % args.render_every == 0:
                rgb = gpu.render(scene, 720, 480, dt=0.0)
                (output / f"{frame:04d}.png").write_bytes(rgb_png(720, 480, rgb))
    finally:
        gpu.close()
        (output / "stats.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
