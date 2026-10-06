import numpy as np
BVH = "/home/ifpt147/motion_capture_data/2026_09_22/squatting.bvh"
CSV = "csv/squatting.csv"

lines = open(BVH).read().splitlines()
m = next(i for i, l in enumerate(lines) if l.strip().startswith("Frame Time"))
bvh = np.array([[float(x) for x in l.split()] for l in lines[m+1:m+8]])
csv = np.loadtxt(CSV, delimiter=",")[:7]     # cols: root xyz, quat xyzw, 28 joints

def step(x): return np.linalg.norm(np.diff(x, axis=0), axis=1).round(3)
print("BVH frames:", len(lines) - m - 1, "  CSV frames:", len(np.loadtxt(CSV, delimiter=",")))
print("BVH root step (m)        :", step(bvh[:, :3] * 0.01))
print("BVH max channel step(deg):", np.abs(np.diff(bvh[:, 3:], axis=0)).max(1).round(2))
print("CSV root step (m)        :", step(csv[:, :3]))
print("CSV quat step            :", step(csv[:, 3:7]))
print("CSV max joint step (rad) :", np.abs(np.diff(csv[:, 7:], axis=0)).max(1).round(3))
print("CSV root pos frames 0-2  :", csv[:3, :3].round(3))
