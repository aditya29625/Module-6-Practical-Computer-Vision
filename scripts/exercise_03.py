"""
Exercise 03 — Horizontal Edge Detector
========================================
Data:        Same synthetic image as Exercise 02
Program:     Apply horizontal edge kernel via cv2.filter2D
Observation: Horizontal edges (top-to-bottom transitions) are highlighted.
What Learned: Rotating the kernel by 90° detects perpendicular edges.
Practical Use: Road-lane detection, horizon detection.
Limitation:  Only detects horizontal edges; diagonal edges are missed.
Next Question: Can we combine both kernels to detect edges in all directions?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_03")
os.makedirs(OUT, exist_ok=True)

def make_test_image():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[40:90, :] = 200   # bright horizontal band
    img[:, 30:80] = np.maximum(img[:, 30:80], 100)
    return img

img = make_test_image()

# ── Kernels ──────────────────────────────────────────────────────────────────
vertical_kernel = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
], dtype=np.float32)

horizontal_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
], dtype=np.float32)

print("=" * 55)
print("EXERCISE 03 — HORIZONTAL EDGE DETECTOR")
print("=" * 55)
print(f"\nHorizontal kernel:\n{horizontal_kernel}")
print(f"\nDifference from vertical kernel: rows are [-1,-1,-1] / [0,0,0] / [1,1,1]")
print("instead of columns being [-1,0,1]")

v_resp = np.abs(cv2.filter2D(img.astype(np.float32), -1, vertical_kernel)).astype(np.uint8)
h_resp = np.abs(cv2.filter2D(img.astype(np.float32), -1, horizontal_kernel)).astype(np.uint8)

print(f"\nVertical response   — non-zero pixels: {np.count_nonzero(v_resp)}")
print(f"Horizontal response — non-zero pixels: {np.count_nonzero(h_resp)}")

# ── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 03 — Horizontal vs Vertical Edge Detector", fontsize=13, fontweight="bold")

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

im1 = axes[1].imshow(vertical_kernel, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_title("Vertical kernel")
for r in range(3):
    for c in range(3):
        axes[1].text(c, r, f"{int(vertical_kernel[r,c])}", ha="center", va="center", fontsize=11)

im2 = axes[2].imshow(horizontal_kernel, cmap="RdBu_r", vmin=-1, vmax=1)
axes[2].set_title("Horizontal kernel")
for r in range(3):
    for c in range(3):
        axes[2].text(c, r, f"{int(horizontal_kernel[r,c])}", ha="center", va="center", fontsize=11)

axes[3].imshow(np.hstack([v_resp, np.ones((128, 4), dtype=np.uint8)*128, h_resp]), cmap="hot")
axes[3].set_title("Left: vertical response\nRight: horizontal response")
axes[3].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_03_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
