#!/bin/sh
# Build locally, then install the renderer family with a rollback copy.
# Deliberately does not change caf-lid-guard, shell settings, or power settings.
set -eu
cd "$(dirname "$0")"
prefix="${CAF_INSTALL_PREFIX:-$HOME/.local/bin}"
mkdir -p .build
xcrun swiftc -O -module-cache-path "$PWD/.build/module-cache" native/caf-metal.swift -o .build/caf-metal
# A sandbox or headless Mac may lack a Metal device; runtime falls back safely.
mkdir -p "$prefix"
backup=$(mktemp -d "$prefix/.caf-backup-XXXXXXXX")
for file in caf-art caf_graphics.py caf_motion.py caf-metal caf.metal caf-fluid.metal caf-scene.metal caf-detail.metal; do
    if [ -f "$prefix/$file" ]; then cp -p "$prefix/$file" "$backup/$file"; fi
done
install -m 644 caf_graphics.py caf_motion.py native/caf.metal native/caf-fluid.metal native/caf-scene.metal native/caf-detail.metal "$prefix/"
install -m 755 .build/caf-metal "$prefix/caf-metal"
install -m 755 caf-art "$prefix/caf-art"
printf 'Installed caf graphics in %s\nPrevious files: %s\nRun caf; use caf --calibrate to teach the two tilt directions.\n' "$prefix" "$backup"
