#!/usr/bin/env python3
"""Actual coffee spray, film/stain aging and conservation probe; no sensors/power."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_graphics import MetalRenderer, rgb_png  # noqa: E402 — local source import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--output", default=str(ROOT / ".build/detail-probe"))
    ap.add_argument("--cream", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    s = runpy.run_path(str(ROOT / "caf-art"), run_name="detail_probe")["Scene"]("diner")
    s.real_fluid = s.mode3d = True
    s.layout(90, 31)
    s.yaw = math.pi / 2
    s.heat = 0.1
    s.ambient = 0.65
    gpu = MetalRenderer(ROOT / "caf_graphics.py")
    records = []
    capture = {
        0.0,
        1.0,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.7,
        3.0,
        4.0,
        8.0,
        15.0,
        30.0,
        60.0,
        90.0,
        120.0,
        180.0,
    }
    t = 0.0
    previous_t = 0.0
    frame = 0
    try:
        while t <= args.seconds + 0.0001:
            dt = t - previous_t
            s.t = t
            if frame == 10 and args.cream:
                s.add_cream()
            if abs(t - 2.0) < 0.0001:
                s.knock_over()
            st = gpu.advance(s, dt)
            assert st["film_balance_error"] == 0, st
            assert (
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
                )
                == st["allocated"]
            ), st
            assert st["invalid"] == 0 and st["body_invalid"] == 0, st
            records.append(dict(st, time=t))
            if any(abs(t - target) < 0.0001 for target in capture):
                (out / f"{t:06.2f}.png").write_bytes(
                    rgb_png(720, 480, gpu.render(s, 720, 480, dt=0))
                )
                print(json.dumps(records[-1]), flush=True)
            previous_t = t
            t = round(t + (0.01 if 2.0 <= t < 3.0 else 0.1), 5)
            frame += 1
    finally:
        gpu.close()
    (out / "stats.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
