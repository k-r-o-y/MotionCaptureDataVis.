# --- 3D Skeleton Visualization (Enhanced Cinematic Version) ---

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ==============================================================
# 1️⃣  Demo Data (if LJ, RJ, Root not already defined)
# ==============================================================

if "LJ" not in locals() or "RJ" not in locals() or "Root" not in locals():
    print("⚙️  Generating synthetic demo motion data...")
    frames = 300
    t = np.linspace(0, 2*np.pi, frames)

    # Root follows a figure-8
    Root = np.stack([
        0.4*np.sin(1.2*t),
        -1.0 + 0.05*np.sin(2*t + 0.5),
        0.3*np.sin(0.8*t)*np.cos(t)
    ], axis=1)

    def chain(side_sign):
        chain = np.zeros((frames, 5, 3))
        for f in range(frames):
            base = Root[f]
            for j in range(5):
                angle = t[f]*2 + j*0.5*side_sign
                x = base[0] + side_sign*(0.15*j)*np.cos(angle)
                y = base[1] - 0.1*j + 0.03*np.sin(angle*3)
                z = base[2] + (0.15*j)*np.sin(angle)
                chain[f,j] = [x,y,z]
        return chain

    LJ, RJ = chain(-1), chain(+1)

# ==============================================================
# 2️⃣  Helper Functions
# ==============================================================

def cube_limits(xs, ys, zs, pad=0.15):
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    r = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.5 * (1 + pad)
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def make_3d_trail(ax, cmap="plasma"):
    lc = Line3DCollection([], cmap=cmap, linewidths=3.5, alpha=0.95)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    lc.set_norm(norm)
    ax.add_collection(lc)
    return lc, norm

def update_trail(lc, norm, path, i, trail=90):
    """Update the neon trails with color by velocity magnitude."""
    j0 = max(0, i-trail)
    seg = np.stack([path[j0:i], path[j0+1:i+1]], axis=1) if i>0 else np.empty((0,2,3))
    if seg.shape[0] == 0:
        lc.set_segments([])
        return
    spd = np.linalg.norm(path[j0+1:i+1] - path[j0:i], axis=1)
    lo, hi = np.percentile(spd, [5,95])
    norm.vmin, norm.vmax = lo, hi
    lc.set_array(spd)
    lc.set_segments(seg)

# ==============================================================
# 3️⃣  Figure Setup
# ==============================================================

F = LJ.shape[0]
allX = np.r_[LJ[:,:,0].ravel(), RJ[:,:,0].ravel(), Root[:,0]]
allY = np.r_[LJ[:,:,1].ravel(), RJ[:,:,1].ravel(), Root[:,1]]
allZ = np.r_[LJ[:,:,2].ravel(), RJ[:,:,2].ravel(), Root[:,2]]
(xlo, xhi), (ylo, yhi), (zlo, zhi) = cube_limits(allX, allY, allZ, pad=0.25)

plt.style.use("dark_background")
fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Cinematic 3D Skeleton — Neon Motion Trails", fontsize=14, pad=20)

# Equal cube bounds
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.set_zlim(zlo, zhi)
ax.invert_yaxis()

# Soft ground plane (slightly lifted)
gy = np.median(allY) + 0.05
gx = np.linspace(xlo, xhi, 25)
gz = np.linspace(zlo, zhi, 25)
GX, GZ = np.meshgrid(gx, gz)
GY = np.full_like(GX, gy)
ax.plot_surface(GX, GY, GZ, alpha=0.12, color="#00FFAA", edgecolor="none")

# Glow + skeleton lines
glowL, = ax.plot([], [], [], lw=10, alpha=0.09, color="#00FFFF")
glowR, = ax.plot([], [], [], lw=10, alpha=0.09, color="#FF55FF")
lineL, = ax.plot([], [], [], lw=2.8, marker="o", markersize=5, color="#00FFFF", label="Left Chain")
lineR, = ax.plot([], [], [], lw=2.8, marker="o", markersize=5, color="#FF55FF", label="Right Chain")
root3d, = ax.plot([], [], [], "--", lw=1.5, alpha=0.7, color="white", label="Root Path")

# Neon trails
trailL, normL = make_3d_trail(ax, cmap="winter")
trailR, normR = make_3d_trail(ax, cmap="autumn")

# HUD text
txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=11, color="white")
ax.legend(loc="upper left", fontsize=10)

# Softer grid
ax.grid(True, color="gray", alpha=0.25)

# ==============================================================
# 4️⃣  Animation
# ==============================================================

FPS = 30
left_tip, right_tip = LJ[:,-1,:], RJ[:,-1,:]

def init3d():
    for ln in (glowL, glowR, lineL, lineR, root3d):
        ln.set_data([], []); ln.set_3d_properties([])
    trailL.set_segments([]); trailR.set_segments([])
    txt.set_text("")
    return glowL, glowR, lineL, lineR, root3d, trailL, trailR, txt

def upd3d(i):
    Lp, Rp = LJ[i], RJ[i]

    # Left + right chain updates
    for ln in (glowL, lineL):
        ln.set_data(Lp[:,0], Lp[:,1]); ln.set_3d_properties(Lp[:,2])
    for ln in (glowR, lineR):
        ln.set_data(Rp[:,0], Rp[:,1]); ln.set_3d_properties(Rp[:,2])

    # Root path + trails
    root3d.set_data(Root[:i+1,0], Root[:i+1,1])
    root3d.set_3d_properties(Root[:i+1,2])
    update_trail(trailL, normL, left_tip, i, trail=90)
    update_trail(trailR, normR, right_tip, i, trail=90)

    # Camera movement (slow cinematic orbit)
    az = 45 + 15 * np.sin(i * 2 * np.pi / (F * 1.8))
    el = 22 + 5 * np.sin(i * 2 * np.pi / (F * 3))
    ax.view_init(elev=el, azim=az)

    # Frame HUD
    txt.set_text(f"Frame {i+1}/{F}")
    return glowL, glowR, lineL, lineR, root3d, trailL, trailR, txt

anim3d = FuncAnimation(fig, upd3d, init_func=init3d,
                       frames=F, interval=1000//FPS, blit=False)

plt.tight_layout()
plt.show()
