"""
Exercise 08 — Inspect RGB Channels
====================================
Data:        Synthetic RGB image with distinct colour regions
Program:     Load RGB image, inspect shape/dtype, split into R/G/B channels.
Observation: Each channel is a 2D grayscale array; colour arises from their combination.
What Learned: RGB images are 3D arrays (H, W, 3); each channel encodes one colour component.
Practical Use: Colour-based segmentation, white-balance correction, channel analysis.
Limitation:  RGB mixes colour and brightness — HSV or LAB are better for colour analysis.
Next Question: Is there a representation that separates colour from brightness?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_08")
os.makedirs(OUT, exist_ok=True)

# Synthetic RGB image
h, w = 128, 128
rgb = np.zeros((h, w, 3), dtype=np.uint8)
rgb[0:43,  :, 0] = 220   # top third → RED
rgb[43:86, :, 1] = 220   # middle third → GREEN
rgb[86:,   :, 2] = 220   # bottom third → BLUE
# Diagonal blend
for i in range(h):
    rgb[i, :, 0] = np.maximum(rgb[i, :, 0], np.linspace(0, 80, w).astype(np.uint8))

print("=" * 55)
print("EXERCISE 08 — INSPECT RGB CHANNELS")
print("=" * 55)
print(f"\nImage shape : {rgb.shape}  → (H={h}, W={w}, C=3)")
print(f"Image dtype : {rgb.dtype}")
print(f"Pixel at (0, 0)  : R={rgb[0,0,0]}, G={rgb[0,0,1]}, B={rgb[0,0,2]}")
print(f"Pixel at (64, 64): R={rgb[64,64,0]}, G={rgb[64,64,1]}, B={rgb[64,64,2]}")
print(f"Pixel at (100,64): R={rgb[100,64,0]}, G={rgb[100,64,1]}, B={rgb[100,64,2]}")

R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
for ch_name, ch in [("R", R), ("G", G), ("B", B)]:
    print(f"\n  {ch_name} channel — shape: {ch.shape}  min: {ch.min()}  max: {ch.max()}  mean: {ch.mean():.1f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 08 — RGB Channel Inspection", fontsize=13, fontweight="bold")

axes[0].imshow(rgb);          axes[0].set_title("RGB composite"); axes[0].axis("off")
axes[1].imshow(R, cmap="Reds_r");   axes[1].set_title("R channel"); axes[1].axis("off")
axes[2].imshow(G, cmap="Greens_r"); axes[2].set_title("G channel"); axes[2].axis("off")
axes[3].imshow(B, cmap="Blues_r");  axes[3].set_title("B channel"); axes[3].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_08_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
