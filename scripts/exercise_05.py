"""
Exercise 05 — Blur with Averaging Kernel
=========================================
Data:        Noisy synthetic image
Program:     Apply 5x5 averaging kernel: blur_kernel = np.ones((5,5)) / 25
Observation: Noise is reduced but edges become blurry (smoothed).
What Learned: Averaging replaces each pixel with the mean of its neighbourhood.
Practical Use: Noise reduction before edge detection or segmentation.
Limitation:  Blurs edges — not edge-preserving. Larger kernel = more blur.
Next Question: Is there a smarter blur that preserves edges better?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_05")
os.makedirs(OUT, exist_ok=True)

# Synthetic noisy image
rng = np.random.default_rng(42)
base = np.zeros((128, 128), dtype=np.float32)
base[30:90, 30:90] = 200
base[50:70, 10:118] = 150
noise = rng.normal(0, 30, base.shape)
noisy = np.clip(base + noise, 0, 255).astype(np.uint8)

# Averaging kernel
blur_kernel = np.ones((5, 5)) / 25

print("=" * 55)
print("EXERCISE 05 — BLUR WITH AVERAGING KERNEL")
print("=" * 55)
print(f"\nKernel (5×5 averaging):")
print(blur_kernel)
print(f"\nKernel sum: {blur_kernel.sum():.2f}  (must be 1.0 for energy-preserving blur)")

# Apply blur
blurred = cv2.filter2D(noisy, -1, blur_kernel.astype(np.float32))

noise_original = np.std(noisy.astype(float))
noise_blurred  = np.std(blurred.astype(float))
print(f"\nOriginal std (proxy for noise): {noise_original:.2f}")
print(f"Blurred  std (proxy for noise): {noise_blurred:.2f}")
print(f"Noise reduction: {(1 - noise_blurred/noise_original)*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Exercise 05 — 5×5 Averaging Blur", fontsize=13, fontweight="bold")
axes[0].imshow(noisy, cmap="gray");   axes[0].set_title("Original (noisy)");  axes[0].axis("off")
axes[1].imshow(blurred, cmap="gray"); axes[1].set_title("Blurred (5×5 avg)"); axes[1].axis("off")
diff = np.abs(noisy.astype(int) - blurred.astype(int)).astype(np.uint8)
axes[2].imshow(diff, cmap="hot");     axes[2].set_title("Difference (removed)"); axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_05_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
