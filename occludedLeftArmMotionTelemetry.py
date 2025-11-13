# ==========================================================
# FULL OCCLUSION PATH VISUALIZATION + ANIMATION
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

sns.set_theme(context="talk", style="whitegrid")

# ==========================================================
# 1) SYNTHETIC MOTION MATCHING YOUR DESCRIPTION
#    - start at centre
#    - walk in a circle
#    - return to centre
#    - do one jumping jack
# ==========================================================

F_circle = 240          # frames for circular walk
F_jump   = 60           # frames for jumping jack
F        = F_circle + F_jump
frames   = np.arange(F)

# --- root path (X,Z) ---
theta = np.linspace(0, 2*np.pi, F_circle, endpoint=False)
radius = 2.0

# circular section
root_circle_x = radius * np.cos(theta)
root_circle_z = radius * np.sin(theta)

# back to centre for jump: hold at origin
root_jump_x = np.zeros(F_jump)
root_jump_z = np.zeros(F_jump)

# combine
root_x = np.concatenate([root_circle_x, root_jump_x])
root_z = np.concatenate([root_circle_z, root_jump_z])

# pack into Root (Y just 0 for this viz)
Root = np.stack([root_x, np.zeros_like(root_x), root_z], axis=1)

# ==========================================================
# 2) SIMPLE LEFT-ARM OCCLUSION MODEL
#    - mostly visible on front half of circle
#    - heavily occluded on back half
#    - visible again during jumping jack
# ==========================================================

left_visible_ratio = np.ones(F)

half_circle = F_circle // 2

# strong occlusions on second half of circle
left_visible_ratio[half_circle:F_circle] = 0.4   # only ~40% markers tracked

# add a bit of noise so it looks less synthetic
left_visible_ratio[:F_circle] += 0.05 * np.random.randn(F_circle)
left_visible_ratio = np.clip(left_visible_ratio, 0.0, 1.0)

# during jumping jack: high visibility
left_visible_ratio[F_circle:] = 0.95 + 0.03 * np.random.randn(F_jump)
left_visible_ratio = np.clip(left_visible_ratio, 0.0, 1.0)

# treat anything below 95% as "occluded"
is_occluded = left_visible_ratio < 0.95

# ==========================================================
# 3) STATIC FIGURE: PATH + OCCLUSION TIMELINE
# ==========================================================

fig, (ax_path, ax_timeline) = plt.subplots(
    2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1]}
)

# --- Top-down motion path coloured by time ---
t_norm = frames / (F - 1)
sc = ax_path.scatter(
    Root[:, 0], Root[:, 2],
    c=t_norm, cmap="viridis",
    s=30, alpha=0.9, edgecolors="none"
)
ax_path.plot(Root[:, 0], Root[:, 2], color="grey", alpha=0.5, linewidth=1)

# highlight occluded frames
ax_path.scatter(
    Root[is_occluded, 0],
    Root[is_occluded, 2],
    facecolors="none",
    edgecolors="red",
    linewidths=1.8,
    s=90,
    label="Left arm occluded"
)

# mark start and end (jumping jack at end)
ax_path.scatter(
    Root[0, 0], Root[0, 2],
    marker="*", s=140, color="white", edgecolors="black",
    zorder=5, label="Start"
)
ax_path.scatter(
    Root[-1, 0], Root[-1, 2],
    marker="^", s=140, color="yellow", edgecolors="black",
    zorder=5, label="End (jumping jack)"
)

ax_path.set_aspect("equal", "box")
ax_path.set_xlabel("X (sideways)")
ax_path.set_ylabel("Z (forward)")
ax_path.set_title("Top-Down Motion Path with Left-Arm Occlusions")

cbar = fig.colorbar(sc, ax=ax_path, pad=0.01)
cbar.set_label("Normalised time (0 = start, 1 = end)")
ax_path.legend(loc="upper left")

# --- Left-arm visibility timeline ---
ax_timeline.plot(frames, left_visible_ratio, color="black", linewidth=2)
ax_timeline.fill_between(
    frames, 0, left_visible_ratio,
    where=is_occluded,
    color="red", alpha=0.25,
    label="Left arm occluded"
)

ax_timeline.set_ylim(-0.05, 1.05)
ax_timeline.set_xlabel("Frame")
ax_timeline.set_ylabel("Left arm visibility\n(fraction of markers)")
ax_timeline.set_title("Left-Arm Occlusion Timeline")

# highlight the jumping-jack segment
ax_timeline.axvspan(
    F_circle, F - 1,
    color="gold", alpha=0.15,
    label="Jumping jack segment"
)

ax_timeline.legend(loc="lower right")

plt.tight_layout()
plt.show()

# ==========================================================
# 4) ANIMATED FIGURE: MOVING MARKER + TIMELINE CURSOR
# ==========================================================

fig_anim, (axA, axB) = plt.subplots(
    2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1]}
)

# --- Background for path (lighter than static for contrast) ---
scA = axA.scatter(
    Root[:, 0], Root[:, 2],
    c=t_norm, cmap="viridis",
    s=20, alpha=0.4, edgecolors="none"
)
axA.plot(Root[:, 0], Root[:, 2], color="grey", alpha=0.3, linewidth=1)

# background occlusion rings
axA.scatter(
    Root[is_occluded, 0],
    Root[is_occluded, 2],
    facecolors="none",
    edgecolors="red",
    linewidths=1.0,
    s=60,
    alpha=0.4,
    label="Left arm occluded (any time)"
)

axA.scatter(
    Root[0, 0], Root[0, 2],
    marker="*", s=120, color="white", edgecolors="black", zorder=5,
    label="Start"
)
axA.scatter(
    Root[-1, 0], Root[-1, 2],
    marker="^", s=120, color="yellow", edgecolors="black", zorder=5,
    label="End (jumping jack)"
)

axA.set_aspect("equal", "box")
axA.set_xlabel("X (sideways)")
axA.set_ylabel("Z (forward)")
axA.set_title("Animated Motion Path with Live Left-Arm Occlusion")

cbarA = fig_anim.colorbar(scA, ax=axA, pad=0.01)
cbarA.set_label("Normalised time (0 = start, 1 = end)")
axA.legend(loc="upper left")

# --- Background timeline ---
axB.plot(frames, left_visible_ratio, color="black", linewidth=1.5, alpha=0.7)
axB.fill_between(
    frames, 0, left_visible_ratio,
    where=is_occluded,
    color="red", alpha=0.2,
    label="Left arm occluded"
)
axB.set_ylim(-0.05, 1.05)
axB.set_xlabel("Frame")
axB.set_ylabel("Left arm visibility\n(fraction of markers)")
axB.set_title("Left-Arm Occlusion Timeline (Live Cursor)")
axB.axvspan(
    F_circle, F - 1,
    color="gold", alpha=0.15,
    label="Jumping jack segment"
)
axB.legend(loc="lower right")

# --- Animated elements ---
current_dot, = axA.plot([], [], marker="o", markersize=10,
                        markeredgewidth=2, linestyle="none")
time_cursor = axB.axvline(frames[0], color="blue", linewidth=2)
status_text = axA.text(
    0.02, 0.95, "", transform=axA.transAxes,
    va="top", ha="left", fontsize=11,
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
)

def init_anim():
    current_dot.set_data([], [])
    time_cursor.set_xdata(frames[0])
    status_text.set_text("")
    return current_dot, time_cursor, status_text

def update_anim(i):
    x, z = Root[i, 0], Root[i, 2]

    if is_occluded[i]:
        face = "red"
        edge = "darkred"
        label = "LEFT ARM OCCLUDED"
    else:
        face = "limegreen"
        edge = "darkgreen"
        label = "Left arm visible"

    current_dot.set_data([x], [z])
    current_dot.set_markerfacecolor(face)
    current_dot.set_markeredgecolor(edge)

    time_cursor.set_xdata(frames[i])

    status_text.set_text(
        f"Frame {i}/{F-1}  |  {label}  |  visibility={left_visible_ratio[i]:.2f}"
    )

    return current_dot, time_cursor, status_text

anim = FuncAnimation(
    fig_anim, update_anim, init_func=init_anim,
    frames=F, interval=40, blit=False
)

plt.tight_layout()
plt.show()

# Optional saving:
# anim.save("motion_path_left_arm_occlusion.mp4", dpi=150, fps=25)
# anim.save("motion_path_left_arm_occlusion.gif", dpi=120, fps=25, writer="pillow")
