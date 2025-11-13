# ==========================================================
# 3D TELEMETRY MOTION: CIRCLE + JUMPING JACK
# WITH OCCLUDED LEFT ARM
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------------------------------------
# 1) Synthetic motion: root + left-hand trajectory
# ----------------------------------------------------------

F_circle = 240          # frames for circular walk
F_jump   = 60           # frames for jumping jack
F        = F_circle + F_jump
frames   = np.arange(F)

theta  = np.linspace(0, 2*np.pi, F_circle, endpoint=False)
radius = 2.0

# Root path in XZ (character walking around centre)
root_circle_x = radius * np.cos(theta)
root_circle_z = radius * np.sin(theta)

# Jumping jack at centre (small vertical bounce)
root_jump_x = np.zeros(F_jump)
root_jump_z = np.zeros(F_jump)
jump_phase  = np.linspace(0, np.pi, F_jump)   # one hop
root_jump_y = 0.0 + 0.3 * np.sin(jump_phase) # vertical hop

root_x = np.concatenate([root_circle_x, root_jump_x])
root_z = np.concatenate([root_circle_z, root_jump_z])
root_y = np.concatenate([np.zeros(F_circle), root_jump_y])

Root = np.stack([root_x, root_y, root_z], axis=1)  # (F,3)

# Left hand: simple model:
# - while walking: left hand offset from root, swinging a bit
# - while jumping: hands go from down to up (like a jack)

left_hand = np.zeros_like(Root)

# walking segment
swing_phase = np.linspace(0, 4*np.pi, F_circle)  # faster swing than step cycle
for i in range(F_circle):
    base = Root[i]
    # offset in local space (very approximate)
    off_x = -0.6                                # left from body
    off_y = 0.8 + 0.2 * np.sin(swing_phase[i])  # small up/down swing
    off_z = 0.3 * np.cos(swing_phase[i])        # tiny forward/back swing
    left_hand[i] = base + np.array([off_x, off_y, off_z])

# jumping-jack segment: arms move upward
for j in range(F_jump):
    i = F_circle + j
    base = Root[i]
    t_rel = j / (F_jump - 1 + 1e-8)  # 0→1
    # arms go from down (~0.5) to up (~1.6)
    off_x = -0.8
    off_y = 0.5 + 1.1 * t_rel
    off_z = 0.0
    left_hand[i] = base + np.array([off_x, off_y, off_z])

# ----------------------------------------------------------
# 2) Occlusion model for left arm
#    - assume sensors roughly along +Z axis
#    - when subject is on "back half" of circle, left arm is occluded
# ----------------------------------------------------------

left_visible_ratio = np.ones(F)

half_circle = F_circle // 2
# occluded during back half of circle
left_visible_ratio[half_circle:F_circle] = 0.3 + 0.1*np.random.randn(F_circle-half_circle)
left_visible_ratio = np.clip(left_visible_ratio, 0.0, 1.0)

# jumping jack visible again
left_visible_ratio[F_circle:] = 0.95 + 0.03*np.random.randn(F_jump)
left_visible_ratio = np.clip(left_visible_ratio, 0.0, 1.0)

is_occluded = left_visible_ratio < 0.8  # visibility < 80% ⇒ occluded-ish

# phases for text readout
phase = np.empty(F, dtype=object)
phase[:half_circle]        = "Circle (front / facing sensors)"
phase[half_circle:F_circle]= "Circle (back / occluded side)"
phase[F_circle:]           = "Jumping jack at centre"

# ----------------------------------------------------------
# 3) 3D figure setup
# ----------------------------------------------------------

plt.style.use("dark_background")
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("3D Telemetry Motion — Circle + Jumping Jack with Left-Arm Occlusion")

# bounds (cube so it doesn't squash)
allX = np.r_[Root[:,0], left_hand[:,0]]
allY = np.r_[Root[:,1], left_hand[:,1]]
allZ = np.r_[Root[:,2], left_hand[:,2]]
xmin,xmax = allX.min(), allX.max()
ymin,ymax = allY.min(), allY.max()
zmin,zmax = allZ.min(), allZ.max()
cx,cy,cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
r = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.6
ax.set_xlim(cx-r, cx+r)
ax.set_ylim(cy-r, cy+r)
ax.set_zlim(cz-r, cz+r)
ax.invert_yaxis()  # mocap-style Y-down if you want; comment out if confusing

# soft ground plane
gx = np.linspace(cx-r, cx+r, 30)
gz = np.linspace(cz-r, cz+r, 30)
GX, GZ = np.meshgrid(gx, gz)
GY = np.full_like(GX, ymin-0.05*(ymax-ymin))
ax.plot_surface(GX, GY, GZ, alpha=0.10, color="#888888", edgecolor="none", zorder=-10)

ax.set_xlabel("X (sideways)")
ax.set_ylabel("Y (up)")
ax.set_zlabel("Z (forward)")

# root trail & marker
root_trail, = ax.plot([], [], [], lw=2, color="white", alpha=0.7, label="Root path")
root_dot,   = ax.plot([], [], [], marker="o", markersize=7, color="white")

# left-hand trail & marker
lh_trail,   = ax.plot([], [], [], lw=2, color="#aaaaaa", alpha=0.6, label="Left hand path")
lh_dot,     = ax.plot([], [], [], marker="o", markersize=9, linestyle="none")

# status text
status = ax.text2D(
    0.02, 0.95, "",
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=10,
    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
)

ax.legend(loc="upper left")

# ----------------------------------------------------------
# 4) Animation functions
# ----------------------------------------------------------

FPS = 30

def init():
    root_trail.set_data([], [])
    root_trail.set_3d_properties([])
    lh_trail.set_data([], [])
    lh_trail.set_3d_properties([])
    root_dot.set_data([], [])
    root_dot.set_3d_properties([])
    lh_dot.set_data([], [])
    lh_dot.set_3d_properties([])
    status.set_text("")
    return root_trail, lh_trail, root_dot, lh_dot, status

def update(i):
    # trails up to frame i
    root_trail.set_data(Root[:i+1,0], Root[:i+1,1])
    root_trail.set_3d_properties(Root[:i+1,2])

    lh_trail.set_data(left_hand[:i+1,0], left_hand[:i+1,1])
    lh_trail.set_3d_properties(left_hand[:i+1,2])

    # current markers
    root_dot.set_data([Root[i,0]], [Root[i,1]])
    root_dot.set_3d_properties([Root[i,2]])

    lh_dot.set_data([left_hand[i,0]], [left_hand[i,1]])
    lh_dot.set_3d_properties([left_hand[i,2]])

    # colour left hand by occlusion
    if is_occluded[i]:
        face, edge, label = "red", "darkred", "LEFT ARM OCCLUDED"
    else:
        face, edge, label = "limegreen", "darkgreen", "Left arm visible"

    lh_dot.set_markerfacecolor(face)
    lh_dot.set_markeredgecolor(edge)

    # camera orbit for telemetry feel
    az = 45 + 35*np.sin(2*np.pi*i / (F*1.3))
    el = 20 + 5*np.sin(2*np.pi*i / (F*2.0))
    ax.view_init(elev=el, azim=az)

    # status text (telemetry)
    status.set_text(
        f"Frame {i+1:03d}/{F} | {phase[i]}\n"
        f"Left-arm visibility: {left_visible_ratio[i]:.2f}  ({label})"
    )

    return root_trail, lh_trail, root_dot, lh_dot, status

anim = FuncAnimation(
    fig, update, init_func=init,
    frames=F, interval=1000//FPS, blit=False
)

plt.tight_layout()
plt.show()

# Optional saving (if you want a video/gif):
# anim.save("3d_telemetry_circle_jump_leftarm_occlusion.mp4", dpi=150, fps=FPS)
# from matplotlib.animation import PillowWriter
# anim.save("3d_telemetry_circle_jump_leftarm_occlusion.gif", dpi=120, fps=FPS, writer=PillowWriter())
