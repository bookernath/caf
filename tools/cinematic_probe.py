#!/usr/bin/env python3
"""Deterministic GPU feature acceptance, with no real sensors or power changes."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_graphics import MetalRenderer, rgb_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--action", choices=["rest", "knock", "tilt", "cream", "mix"], default="knock"
    )
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--output", default=str(ROOT / ".build/cinematic"))
    ap.add_argument("--render-every", type=int, default=30)
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--angle", type=float, default=55)
    ap.add_argument("--yaw", type=float, default=0.6)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    s = runpy.run_path(str(ROOT / "caf-art"), run_name="probe")["Scene"]("diner")
    s.real_fluid = s.mode3d = True
    s.cup_motion = not args.locked
    s.layout(90, 31)
    s.ambient = 0.65
    s.heat = 0.2
    s.yaw = args.yaw
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    records = []
    try:
        for frame in range(args.frames + 1):
            s.t = frame / 30
            if frame == 60:
                if args.action == "knock":
                    s.fluid_events.append({"kind": "knock"})
                if args.action in ("cream", "mix"):
                    s.fluid_events.append({"kind": "cream"})
            if args.action == "mix" and frame == 120:
                s.stir()
            angle = (
                args.angle
                * min(1, max(0, (frame - 60) / 30))
                * min(1, max(0, (180 - frame) / 30))
                if args.action == "tilt"
                else 0
            )
            s.gravity = (
                math.sin(math.radians(angle)),
                -math.cos(math.radians(angle)),
                0,
            )
            st = gpu.advance(s, 0 if frame == 0 else 1 / 30)
            st["frame"] = frame
            records.append(dict(st))
            if frame % 15 == 0:
                print(
                    json.dumps(
                        {
                            k: st[k]
                            for k in (
                                "frame",
                                "in_cup",
                                "on_table",
                                "off_scene",
                                "max_penetration",
                                "cup_tilt",
                                "cup_position",
                                "cup_impact",
                                "cup_penetration",
                                "body_invalid",
                                "mean_speed2",
                                "gpu_ms",
                            )
                        }
                    ),
                    flush=True,
                )
            if args.render_every and frame % args.render_every == 0:
                rgb = gpu.render(s, 720, 480, dt=0)
                (out / f"{frame:04d}.png").write_bytes(rgb_png(720, 480, rgb))
    finally:
        gpu.close()
        (out / "stats.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
