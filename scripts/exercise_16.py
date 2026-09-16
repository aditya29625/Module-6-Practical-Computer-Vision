"""
Exercise 16 — Random Crop (RandomResizedCrop)
===============================================
Data:        Synthetic pet image
Program:     transforms.RandomResizedCrop(224, scale=(0.5, 1.0))
Observation: A random portion (50–100% area) is cropped and resized to 224x224.
What Learned: Combines cropping and resizing — mimics different zoom/framing.
Practical Use: Standard augmentation in ImageNet training; forces local feature learning.
Limitation:  May crop out the main subject completely if scale is very small.
Next Question: How do we add colour variations on top of spatial augmentations?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_16")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.RandomResizedCrop(224, scale=(0.5, 1.0))

print("=" * 55)
print("EXERCISE 16 — RANDOM CROP (RandomResizedCrop)")
print("=" * 55)
print(f"\nCrop size   : 224x224 (output always 224x224)")
print(f"Scale range : (0.5, 1.0) → 50–100% of image area is cropped")
print(f"Ratio range : default (0.75, 1.33)")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 16 — RandomResizedCrop (5 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 17)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Crop {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_16_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
