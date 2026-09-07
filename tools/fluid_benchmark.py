#!/usr/bin/env python3
"""Warm Python/Metal/PNG timing; excludes terminal transport and display."""

import argparse
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_graphics import MetalRenderer, rgb_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["tilt", "cinematic"], default="cinematic")
    ap.add_argument("--frames", type=int, default=330)
    ap.add_argument("--output")
    args = ap.parse_args()
    s = runpy.run_path(str(ROOT / "caf-art"), run_name="benchmark")["Scene"]("diner")
    s.mode3d = s.real_fluid = True
    s.layout(90, 31)
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    records = []
    try:
        for frame in range(max(20, args.frames)):
            start = time.perf_counter()
            s.t += 1 / 30
            phase = "tilt"
            if args.scenario == "tilt":
                a = math.radians(22) if 45 < frame < 105 else 0
                s.gravity = (math.sin(a), -math.cos(a), 0)
            else:
                if frame == 30:
                    s.add_cream()
                if frame == 100:
                    s.stir()
                if frame == 200:
                    s.knock_over()
                phase = (
                    "rest"
                    if frame < 30
                    else (
                        "cream" if frame < 100 else ("mix" if frame < 200 else "knock")
                    )
                )
            s.step(1 / 30)
            rgb = gpu.render(s, 720, 480, dt=1 / 30)
            png = rgb_png(720, 480, rgb)
            records.append(
                {
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "gpu_ms": gpu.stats["gpu_ms"],
                    "png_bytes": len(png),
                    "phase": phase,
                }
            )
    finally:
        gpu.close()
    result = {}
    for phase in ["all", *dict.fromkeys(r["phase"] for r in records)]:
        group = [r for r in records[10:] if phase == "all" or r["phase"] == phase]
        if not group:
            continue
        result[phase] = {"frames": len(group)}
        for key in ("total_ms", "gpu_ms", "png_bytes"):
            values = sorted(r[key] for r in group)
            result[phase][key] = {
                "median": statistics.median(values),
                "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps({"summary": result, "frames": records}, indent=2)
        )


if __name__ == "__main__":
    main()
