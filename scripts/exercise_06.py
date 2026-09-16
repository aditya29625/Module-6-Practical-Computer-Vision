"""
Exercise 06 — Gaussian Blur + Canny Edge Detection
====================================================
Data:        Synthetic image with shapes
Program:     cv2.GaussianBlur → cv2.Canny pipeline
Observation: Gaussian blur suppresses noise; Canny produces thin, well-localized edges.
What Learned: Canny = Gaussian blur + gradient + non-max suppression + hysteresis thresholding.
Practical Use: Object boundary detection in robotics, autonomous vehicles.
Limitation:  Two thresholds (100, 200) must be tuned per image type.
Next Question: How can we detect corners (not just edges)?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_06")
os.makedirs(OUT, exist_ok=True)

# Synthetic image
img = np.zeros((128, 128), dtype=np.uint8)
cv2.rectangle(img, (15, 15), (60, 60), 200, -1)
cv2.circle(img, (95, 40), 25, 170, -1)
cv2.ellipse(img, (64, 100), (40, 20), 0, 0, 360, 220, -1)
rng = np.random.default_rng(7)
img = np.clip(img.astype(float) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)

print("=" * 55)
print("EXERCISE 06 — GAUSSIAN BLUR + CANNY EDGE DETECTION")
print("=" * 55)

# Pipeline
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges   = cv2.Canny(blurred, 100, 200)

print(f"\nInput shape      : {img.shape}")
print(f"After GaussBlur  : {blurred.shape}  (same size, noise reduced)")
print(f"After Canny      : {edges.shape}   (binary edge map)")
print(f"Edge pixels      : {np.count_nonzero(edges)}")
print(f"\nCanny thresholds: low=100  high=200")
print("  • Pixels with gradient > 200 → definite edge")
print("  • Pixels with gradient 100–200 → edge only if connected to definite edge")
print("  • Pixels with gradient < 100 → suppressed")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Exercise 06 — Gaussian Blur + Canny Pipeline", fontsize=13, fontweight="bold")
axes[0].imshow(img, cmap="gray");     axes[0].set_title("Original (noisy)"); axes[0].axis("off")
axes[1].imshow(blurred, cmap="gray"); axes[1].set_title("After GaussianBlur(5,5)"); axes[1].axis("off")
axes[2].imshow(edges, cmap="gray");   axes[2].set_title(f"Canny edges\n({np.count_nonzero(edges)} edge pixels)"); axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_06_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
