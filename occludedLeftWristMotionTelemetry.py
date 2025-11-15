# ==========================================================
# TELEMETRY: MOTION + OCCLUDED LEFT WRIST
#   - circle walk + jumping jack
#   - left wrist occluded on left side of circle
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

# ----------------------------------------------------------
# 1) Motion: circle walk + jumping jack (same as 3D vis)
# ----------------------------------------------------------
F_circle = 240          # frames for walking around circle
F_jump   = 60           # frames for jumping jack at centre
F        = F_circle + F_jump
frames   = np.arange(F)

theta  = np.linspace(0, 2*np.pi, F_circle, endpoint=False)
radius = 2.0

# Root path (pelvis) in XZ plane
root_circle_x = radius * np.cos(theta)
root_circle_z = radius * np.sin(theta)

root_jump_x = np.zeros(F_jump)
root_jump_z = np.zeros(F_jump)

# Vertical bob while walking; hop during jump
step_period = 40
walk_phase  = 2*np.pi * np.arange(F_circle) / step_period
root_circle_y = 1.0 + 0.08 * np.sin(2 * walk_phase)

jump_phase  = np.linspace(0, np.pi, F_jump)
root_jump_y = 1.0 + 0.45 * np.sin(jump_phase)   # one hop

root_x = np.concatenate([root_circle_x, root_jump_x])
root_y = np.concatenate([root_circle_y, root_jump_y])
root_z = np.concatenate([root_circle_z, root_jump_z])

Root = np.stack([root_x, root_y, root_z], axis=1)

# Root speed in XZ plane
vel = np.zeros((F, 2))
vel[1:, 0] = np.diff(root_x)
vel[1:, 1] = np.diff(root_z)
speed = np.linalg.norm(vel, axis=1)

# ----------------------------------------------------------
# 2) Left-wrist occlusion pattern (match 3D script)
#    - occluded on LEFT side of circle: X < -0.5*radius (after frame 30)
#    - visible elsewhere and during jumping jack
# ----------------------------------------------------------
np.random.seed(1)

left_wrist_vis = np.ones(F)

# indices on left side of the circle (skip first ~30 frames)
idx_left_side = np.where(
    (np.arange(F_circle) > 30) & (root_circle_x < -0.5*radius)
)[0]

# those frames have reduced visibility
left_wrist_vis[idx_left_side] = 0.35 + 0.15 * np.random.randn(len(idx_left_side))

# jumping jack mostly visible
left_wrist_vis[F_circle:] = 0.96 + 0.03 * np.random.randn(F_jump)

# clamp to [0,1]
left_wrist_vis = np.clip(left_wrist_vis, 0.0, 1.0)

lw_occluded = left_wrist_vis < 0.8

# phase / state labels
state = np.zeros(F, dtype=int)      # 0 = normal circle
state[idx_left_side] = 1            # 1 = left-side wrist occluded
state[F_circle:] = 2                # 2 = jumping jack

state_labels = [
    "Circle (wrist visible)",
    "Circle (left side / wrist occluded)",
    "Jumping jack at centre",
]
state_colors = ["#4daf4a", "#e41a1c", "#377eb8"]

# ----------------------------------------------------------
# 3) Telemetry figure
# ----------------------------------------------------------
fig, (ax_path, ax_speed, ax_vis, ax_state) = plt.subplots(
    4, 1, figsize=(12, 13),
    gridspec_kw={"height_ratios": [2.0, 1.2, 1.2, 0.6]},
    sharex=False
)

# ---------- Top-down path ----------
t_norm = frames / (F - 1 + 1e-8)
sc = ax_path.scatter(
    root_x, root_z,
    c=t_norm, cmap="viridis", s=18, alpha=0.4, edgecolors="none"
)
ax_path.set_aspect("equal", "box")
ax_path.set_xlabel("X (sideways)")
ax_path.set_ylabel("Z (forward)")
ax_path.set_title("Top-Down Motion Path with Left-Wrist Occlusions")

# start / end markers
ax_path.scatter(root_x[0], root_z[0],
                marker="*", s=130, color="white", edgecolors="black",
                zorder=5, label="Start")
ax_path.scatter(root_x[-1], root_z[-1],
                marker="^", s=130, color="yellow", edgecolors="black",
                zorder=5, label="End (jumping jack)")

# highlight occluded frames
ax_path.scatter(
    root_x[lw_occluded], root_z[lw_occluded],
    facecolors="none", edgecolors="red", s=80, linewidths=1.8,
    label="Left wrist occluded"
)

cb = fig.colorbar(sc, ax=ax_path, pad=0.01)
cb.set_label("Normalised time (0 = start, 1 = end)")
ax_path.legend(loc="upper left")

# ---------- Speed telemetry ----------
ax_speed.plot(frames, speed, color="#1f78b4", linewidth=2)
ax_speed.set_ylabel("Root speed\n(units/frame)")
ax_speed.set_title("Telemetry: Motion + Left-Wrist Occlusion")
ax_speed.set_xlim(0, F-1)

# shade circle vs jump
ax_speed.axvspan(0, F_circle-1, color="#cccccc", alpha=0.15, label="Circle")
ax_speed.axvspan(F_circle, F-1, color="#ffbf00", alpha=0.15, label="Jumping jack")
ax_speed.legend(loc="upper right")

# ---------- Visibility telemetry ----------
ax_vis.plot(frames, left_wrist_vis, color="black", linewidth=2)
ax_vis.set_xlim(0, F-1)
ax_vis.set_ylim(-0.05, 1.05)
ax_vis.set_ylabel("Left wrist\nvisibility")

ax_vis.fill_between(
    frames, 0, left_wrist_vis,
    where=lw_occluded,
    color="red", alpha=0.25,
    label="Left wrist occluded"
)
ax_vis.legend(loc="upper right")

# ---------- Phase / state strip ----------
state_strip = state[np.newaxis, :]
cmap = plt.matplotlib.colors.ListedColormap(state_colors)
ax_state.imshow(state_strip, aspect="auto", cmap=cmap,
                extent=[0, F-1, 0, 1])
ax_state.set_yticks([])
ax_state.set_xlabel("Frame")

handles = [
    plt.matplotlib.patches.Patch(color=state_colors[i], label=state_labels[i])
    for i in range(3)
]
ax_state.legend(handles=handles, loc="center",
                bbox_to_anchor=(0.5, -0.6), ncol=2, frameon=False)

plt.tight_layout()
plt.show()
