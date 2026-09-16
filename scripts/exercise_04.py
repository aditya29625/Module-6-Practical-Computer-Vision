"""
Exercise 04 — Sobel X and Sobel Y + Gradient Magnitude
=======================================================
Data:        Synthetic test image with shapes
Program:     Apply Sobel X, Sobel Y kernels, compute magnitude = sqrt(gx^2 + gy^2)
Observation: Sobel X highlights vertical edges; Sobel Y highlights horizontal edges.
             Magnitude image shows ALL edges regardless of orientation.
What Learned: Sobel is a noise-resistant edge detector combining gradient in both directions.
Practical Use: Pre-processing for object detection, feature extraction.
Limitation:  Thick edges at boundaries; sensitive to image scale.
Next Question: Can we apply thresholding on the magnitude to get thin, clean edges?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_04")
os.makedirs(OUT, exist_ok=True)

# Synthetic image with various edges
img = np.zeros((128, 128), dtype=np.uint8)
img[20:108, 20:108] = 180          # large square
cv2.circle(img, (64, 64), 30, 255, -1)   # filled circle on top
cv2.line(img, (10, 10), (118, 118), 140, 3)  # diagonal line

print("=" * 55)
print("EXERCISE 04 — SOBEL X, SOBEL Y AND GRADIENT MAGNITUDE")
print("=" * 55)
print(f"\nInput image shape: {img.shape}")

# Apply Sobel
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gx**2 + gy**2)

# Normalise for display
gx_disp = np.abs(gx).astype(np.uint8)
gy_disp = np.abs(gy).astype(np.uint8)
mag_disp = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

print(f"\nSobel X  — min: {gx.min():.1f}  max: {gx.max():.1f}")
print(f"Sobel Y  — min: {gy.min():.1f}  max: {gy.max():.1f}")
print(f"Magnitude — min: {magnitude.min():.1f}  max: {magnitude.max():.1f}")
print(f"\nEdge pixels (mag > 50): {np.sum(magnitude > 50)}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 04 — Sobel X, Y and Gradient Magnitude", fontsize=13, fontweight="bold")
for ax, data, title in zip(axes,
    [img, gx_disp, gy_disp, mag_disp],
    ["Original", "Sobel X (vertical edges)", "Sobel Y (horizontal edges)", "Gradient Magnitude √(Gx²+Gy²)"]):
    ax.imshow(data, cmap="gray" if title != "Gradient Magnitude √(Gx²+Gy²)" else "hot")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_04_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
