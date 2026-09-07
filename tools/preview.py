#!/usr/bin/env python3
"""Render actual GPU frames without sensors, caffeinate, terminal or sleep changes."""

import argparse
from pathlib import Path
import random
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_graphics import MetalRenderer, rgb_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / ".build/diner.png"))
    ap.add_argument("--tilt-x", type=float, default=0)
    ap.add_argument("--tilt-y", type=float, default=0)
    ap.add_argument("--theme", default="diner")
    ap.add_argument("--word", default="")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ambient", type=float, default=0.8)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()
    m = runpy.run_path(str(ROOT / "caf-art"), run_name="caf_preview")
    random.seed(94)
    scene = m["Scene"](args.theme)
    scene.mode3d = True
    scene.layout(90, 31)
    scene.ambient = args.ambient
    scene.level = scene.target_level = 0.85
    for i in range(90):
        scene.t += 1 / 30
        scene.step(1 / 30)
    scene.tilt_in = (args.tilt_x, args.tilt_y)
    scene.roll = args.tilt_x * 0.09
    if args.word:
        scene.write_word(args.word)
        for i in range(30):
            scene.t += 1 / 30
            scene.step(1 / 30)
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    try:
        times = []
        for _ in range(args.frames):
            start = time.perf_counter()
            rgb = gpu.render(scene, args.width, args.height)
            times.append((time.perf_counter() - start) * 1000)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rgb_png(args.width, args.height, rgb))
        print(
            f"{output}: {args.width}x{args.height}; frame ms: "
            + ", ".join(f"{v:.1f}" for v in times)
        )
    finally:
        gpu.close()


if __name__ == "__main__":
    main()
