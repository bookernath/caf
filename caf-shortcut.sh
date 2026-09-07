# caf — keep the Mac awake behind a live-rendered coffee scene.
#
# Install the complete renderer family with `sh install.sh` from https://github.com/bookernath/caf.
# Python 3.11+, no third-party Python packages; the optional Swift/Metal helper
# is built with Xcode Command Line Tools. GPU frames stay inside the terminal.
# A positive Kitty graphics query enables Metal; unsupported terminals use ANSI.
#
# APIC liquid spills over the rim and pools on the table. f = refill, r = reset.
# m = cream, k = strong cup knock, l = cinematic lighting; --sound adds impact foley.
# --cup locked fixes the cup; --studio disables animated lighting.
# --fluid legacy restores the old heightfield. Requires all four Metal shader files.
# Default: Diner / 94 — ivory ceramic, oxblood stripes, walnut, reflections,
# soft shadows, transmitted coffee, wet patches, meniscus and fine retro dithering.
# `caf --ansi` keeps character rendering; `--renderer cpu` bypasses Metal;
# `--smooth` disables dithering; `--2d` retains the spill/rigid-body scene.
# Presets: diner, classic, matcha, cyber, berry, dark. `o` enables auto-orbit.
#
# Run `caf --calibrate` once: level, right edge down, far edge down, held still.
# It saves the physical screen axes to ~/.config/caf/imu.json. Start caf level;
# the reference stays locked after startup, and `z` re-zeros it without losing
# the directional mapping. `--auto-level` restores gradual automatic re-level.
# `caf --imu-tilt` shows both tilt axes. The legacy CAF_IMU_FLIP still works.
#
# Lid close is a hardware sleep path that ignores caffeinate's power
# assertions, so caf-art also starts ~/.local/bin/caf-lid-guard: it needs
# root (`pmset disablesleep 1`) and a watchdog that clears the flag the
# moment caf exits — Ctrl+C, kill -9, closed terminal, all covered.
# Out of the box that's one interactive sudo per caf session. First-time
# setup to make it prompt-free — grants NOPASSWD for exactly
# `pmset {-a|-b|-c} disablesleep {0|1}` and nothing else (run once, from
# wherever you downloaded caf-lid-guard.sudoers):
#
#   sed "s/^YOURUSER/$(id -un)/" caf-lid-guard.sudoers > /tmp/caf-lid-guard.sudoers
#   sudo visudo -c -f /tmp/caf-lid-guard.sudoers \
#     && sudo install -m 0440 -o root -g wheel \
#          /tmp/caf-lid-guard.sudoers /etc/sudoers.d/caf-lid-guard \
#     && rm /tmp/caf-lid-guard.sudoers
#
# Always visudo -c first — a malformed file in /etc/sudoers.d locks you out
# of sudo. Inspect granted commands with `sudo -l`; undo with `sudo rm /etc/sudoers.d/caf-lid-guard`.
#
# `caf --no-lid-guard` skips the guard; `caf-lid-guard status` shows state.
caf() {
  ~/.local/bin/caf-art "$@"
}
