"""
Exercise 02 — Build a Vertical Edge Detector
=============================================
Data:        Synthetic gradient image from Exercise 01
Program:     Apply vertical edge kernel via cv2.filter2D
Observation: Vertical edges (left-to-right transitions) are highlighted.
What Learned: A kernel is a mathematical operator sliding over the image.
Practical Use: Detecting vertical structures, OCR pre-processing.
Limitation:  Sensitive to noise; only detects one edge orientation.
Next Question: How does a horizontal edge kernel differ?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_02")
os.makedirs(OUT, exist_ok=True)

# ── Create synthetic image ───────────────────────────────────────────────────
def make_test_image():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[:, 40:90] = 200   # bright vertical band → creates two vertical edges
    img[30:80, :] = np.maximum(img[30:80, :], 100)
    return img

img = make_test_image()

# ── Vertical edge kernel ─────────────────────────────────────────────────────
vertical_kernel = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
], dtype=np.float32)

print("=" * 55)
print("EXERCISE 02 — VERTICAL EDGE DETECTOR")
print("=" * 55)
print(f"\nKernel:\n{vertical_kernel}")

# Apply kernel
edge_response = cv2.filter2D(img.astype(np.float32), -1, vertical_kernel)
edge_abs = np.abs(edge_response).astype(np.uint8)

print(f"\nInput image shape  : {img.shape}")
print(f"Output shape       : {edge_abs.shape}")
print(f"Output min/max     : {edge_abs.min()} / {edge_abs.max()}")
print(f"Non-zero pixels    : {np.count_nonzero(edge_abs)}")

# ── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Exercise 02 — Vertical Edge Detector", fontsize=13, fontweight="bold")

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("off")

# Kernel heatmap
im = axes[1].imshow(vertical_kernel, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_title("Vertical kernel\n[-1,0,+1] columns")
for r in range(3):
    for c in range(3):
        axes[1].text(c, r, f"{int(vertical_kernel[r,c])}", ha="center", va="center", fontsize=12)
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(edge_abs, cmap="hot")
axes[2].set_title("Vertical edge response\n(absolute values)")
axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_02_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("\n[Summary]")
print("  • The kernel [-1, 0, +1] computes a difference between left and right pixels")
print("  • High positive response = left-to-right brightness increase (left edge)")
print("  • High negative response = right-to-left brightness increase (right edge)")
print("Done ✓")
