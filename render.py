#!/usr/bin/env python3
"""Turn MPI-IO snapshots (raw float64 N x N) into images for the video.
  python3 render.py data/snap N            -> data/snap_00000.png ... (uses matplotlib if installed, else .ppm)
  python3 render.py data/snap N --gif out.gif  (needs pillow)"""
import sys, glob, struct, array
prefix, N = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(prefix + "_*.bin"))
try:
    import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    have_mpl = True
except ImportError:
    have_mpl = False
def ramp(v):                       # simple black-red-yellow-white heat colormap for the fallback writer
    v = max(0.0, min(1.0, v / 100.0)); r = min(1, v * 3); g = min(1, max(0, v * 3 - 1)); b = min(1, max(0, v * 3 - 2))
    return bytes((int(255 * r), int(255 * g), int(255 * b)))
for f in files:
    out = f[:-4]
    if have_mpl:
        a = np.fromfile(f, dtype="<f8").reshape(N, N)
        plt.figure(figsize=(6, 6)); plt.imshow(a, cmap="inferno", vmin=0, vmax=100); plt.colorbar(label="temperature")
        plt.title(f.split("/")[-1]); plt.axis("off"); plt.savefig(out + ".png", dpi=110, bbox_inches="tight"); plt.close()
        print("wrote", out + ".png")
    else:
        d = array.array("d"); d.fromfile(open(f, "rb"), N * N)
        with open(out + ".ppm", "wb") as o:
            o.write(b"P6 %d %d 255\n" % (N, N)); [o.write(ramp(v)) for v in d]
        print("wrote", out + ".ppm")
if len(sys.argv) > 4 and sys.argv[3] == "--gif":
    from PIL import Image
    ims = [Image.open(f[:-4] + ".png") for f in files]; ims[0].save(sys.argv[4], save_all=True, append_images=ims[1:], duration=200, loop=0)
    print("wrote", sys.argv[4])
