"""
Exercise 01 — Inspect Image as Numbers
=======================================
Data:        Synthetically generated gradient image (avoids dataset download dependency)
Program:     Load/create image, convert to NumPy, inspect shape, pixel values, min/max.
Observation: Pixel values are integers 0-255; shape tells us (H, W) or (H, W, 3).
What Learned: Images are just arrays of numbers.
Practical Use: Preprocessing pipeline, debugging data pipelines.
Limitation:  Only works for 8-bit images without modification.
Next Question: How can we detect edges using pixel value differences?
"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_01")
os.makedirs(OUT, exist_ok=True)

# ── Create a synthetic test image (gradient + checkerboard) ──────────────────
def make_synthetic_image(path_gray, path_rgb):
    """Generate deterministic synthetic images so no internet is needed."""
    h, w = 128, 128
    # Grayscale: horizontal gradient
    gray = np.linspace(0, 255, w, dtype=np.uint8)
    gray_img = np.tile(gray, (h, 1))
    Image.fromarray(gray_img, mode="L").save(path_gray)

    # RGB: colourful gradient
    r = np.linspace(0, 255, w, dtype=np.uint8)
    g = np.linspace(255, 0, h, dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.tile(r, (h, 1))          # red channel  → horizontal gradient
    rgb[:, :, 1] = np.tile(g, (w, 1)).T         # green channel → vertical gradient
    rgb[:, :, 2] = 128                           # blue channel  → constant mid-value
    Image.fromarray(rgb, mode="RGB").save(path_rgb)

gray_path = os.path.join(OUT, "synthetic_gray.png")
rgb_path  = os.path.join(OUT, "synthetic_rgb.png")
make_synthetic_image(gray_path, rgb_path)

# ── Load and inspect grayscale ───────────────────────────────────────────────
gray_pil = Image.open(gray_path).convert("L")
gray_arr = np.array(gray_pil)

print("=" * 55)
print("EXERCISE 01 — INSPECT IMAGE AS NUMBERS")
print("=" * 55)
print(f"\n[Grayscale Image]")
print(f"  Shape          : {gray_arr.shape}   (H, W)")
print(f"  dtype          : {gray_arr.dtype}")
print(f"  Min pixel      : {gray_arr.min()}")
print(f"  Max pixel      : {gray_arr.max()}")
print(f"  Total pixels   : {gray_arr.size}")
print(f"  Sample pixels  :\n{gray_arr[:4, :8]}")

# ── Load and inspect RGB ─────────────────────────────────────────────────────
rgb_pil = Image.open(rgb_path).convert("RGB")
rgb_arr = np.array(rgb_pil)

print(f"\n[RGB Image]")
print(f"  Shape          : {rgb_arr.shape}  (H, W, C)")
print(f"  dtype          : {rgb_arr.dtype}")
print(f"  Min pixel      : {rgb_arr.min()}")
print(f"  Max pixel      : {rgb_arr.max()}")
print(f"  Pixel at (0,0) : R={rgb_arr[0,0,0]}, G={rgb_arr[0,0,1]}, B={rgb_arr[0,0,2]}")
print(f"  Pixel at (64,64): R={rgb_arr[64,64,0]}, G={rgb_arr[64,64,1]}, B={rgb_arr[64,64,2]}")

# ── Normalise to [0,1] ───────────────────────────────────────────────────────
norm = gray_arr / 255.0
print(f"\n[After normalising to [0,1]]")
print(f"  Min: {norm.min():.4f}  Max: {norm.max():.4f}")

# ── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Exercise 01 — Inspect Image as Numbers", fontsize=13, fontweight="bold")

axes[0].imshow(gray_arr, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Grayscale\nShape: {gray_arr.shape}\nMin={gray_arr.min()} Max={gray_arr.max()}")
axes[0].axis("off")

axes[1].imshow(rgb_arr)
axes[1].set_title(f"RGB\nShape: {rgb_arr.shape}")
axes[1].axis("off")

# Pixel value histogram
axes[2].hist(gray_arr.ravel(), bins=64, color="steelblue", edgecolor="none")
axes[2].set_title("Grayscale pixel value histogram")
axes[2].set_xlabel("Pixel value (0–255)")
axes[2].set_ylabel("Count")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_01_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("\n[Summary]")
print("  • Grayscale image shape: (H, W)  — 2 dimensions")
print("  • RGB image shape:       (H, W, 3) — 3 dimensions (channels)")
print("  • Pixel values range 0–255 for uint8 images")
print("  • Normalise by dividing by 255 for neural network input")
print("Done ✓")
