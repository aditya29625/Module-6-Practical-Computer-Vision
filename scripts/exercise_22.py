"""
Exercise 22 — Perspective Distortion
=======================================
Data:        Synthetic pet image
Program:     transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
Observation: Image appears as if viewed from a tilted angle — trapezoid warp.
What Learned: Perspective transforms simulate different camera viewpoints.
Practical Use: Autonomous driving (road plane homography), document scanning.
Limitation:  High distortion causes strong geometric deformation — label may become unclear.
Next Question: How can we occlude (hide) part of the image to simulate real-world obstruction?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_22")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)

print("=" * 55)
print("EXERCISE 22 — PERSPECTIVE DISTORTION")
print("=" * 55)
print("RandomPerspective(distortion_scale=0.5, p=1.0)")
print("  distortion_scale=0.5 → moderate warp")
print("  p=1.0 → always applied")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 22 — Random Perspective Distortion", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 29)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_22_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
