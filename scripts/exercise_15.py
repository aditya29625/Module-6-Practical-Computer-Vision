"""
Exercise 15 — Scale (RandomAffine with scale)
===============================================
Data:        Synthetic pet image
Program:     transforms.RandomAffine(degrees=0, scale=(0.7, 1.3))
Observation: Image appears zoomed in or out while keeping the same output resolution.
What Learned: Scale augmentation simulates different distances from camera.
Practical Use: Object detection where objects appear at varying distances.
Limitation:  Extreme scaling (very small) may lose too much detail.
Next Question: How does RandomResizedCrop combine scaling and cropping?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_15")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.RandomAffine(degrees=0, scale=(0.7, 1.3))

print("=" * 55)
print("EXERCISE 15 — SCALE (RandomAffine with scale)")
print("=" * 55)
print(f"\nScale range   : (0.7, 1.3) → 70% to 130% of original size")
print(f"Output size   : same as input {img.size} — content is zoomed in/out")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 15 — Scale Augmentation (4 random samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 13)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_15_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
