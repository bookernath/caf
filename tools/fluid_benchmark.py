#!/usr/bin/env python3
"""Warm Python/Metal/encode timing per frame, up to the Kitty escape string.

Excludes the terminal's own decode and display. --transport compares PNG
(deflate + base64 through the pty) with POSIX shared memory (t=s: zero-copy,
the terminal reads the pixels); the shm object is unlinked after each frame,
as a terminal does after reading it. --pipeline on overlaps GPU frame N+1
with encoding frame N, as the interactive loop does when a frame would miss
its budget."""

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
from caf_graphics import MetalRenderer, kitty_frame  # noqa: E402


def run(args, width, height, transport, pipeline):
    s = runpy.run_path(str(ROOT / "caf-art"), run_name="benchmark")["Scene"]("diner")
    s.mode3d = s.real_fluid = True
    s.layout(90, 31)
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    gpu.transport, gpu.pipeline = transport, pipeline
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
            result = gpu.render(s, width, height, dt=1 / 30)
            escape = kitty_frame(result, width, height, 90, 30)
            if getattr(result, "shm", None):
                gpu.frames.unlink(result.shm)  # what the terminal does after reading
            if gpu.transport != transport:
                raise SystemExit(f"helper fell back from {transport}")
            records.append(
                {
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "gpu_ms": gpu.stats.get("gpu_ms", 0),
                    "escape_bytes": len(escape),
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
        for key in ("total_ms", "gpu_ms", "escape_bytes"):
            values = sorted(r[key] for r in group)
            result[phase][key] = {
                "median": statistics.median(values),
                "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            }
    return result, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["tilt", "cinematic"], default="cinematic")
    ap.add_argument("--frames", type=int, default=330)
    ap.add_argument("--size", default="720x480", help="comma-separated WxH list")
    ap.add_argument("--transport", default="png", help="comma-separated: png,shm")
    ap.add_argument("--pipeline", choices=["off", "on"], default="off")
    ap.add_argument("--output")
    args = ap.parse_args()
    out, frames = {}, {}
    for size in args.size.split(","):
        width, height = map(int, size.split("x"))
        for transport in args.transport.split(","):
            key = f"{size} {transport} pipeline={args.pipeline}"
            out[key], frames[key] = run(args, width, height, transport, args.pipeline)
    single = len(out) == 1
    print(json.dumps(next(iter(out.values())) if single else out, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps({"summary": out, "frames": frames}, indent=2)
        )


if __name__ == "__main__":
    main()
