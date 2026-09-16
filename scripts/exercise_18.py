"""
Exercise 18 — Contrast Augmentation
======================================
Data:        Synthetic pet image
Program:     transforms.ColorJitter(contrast=0.5)
Observation: Low contrast → flat grey look; high contrast → vivid darks/lights.
What Learned: Contrast is the ratio between darkest and brightest pixels.
Practical Use: Simulates foggy vs crisp imaging conditions.
Limitation:  Extreme low contrast loses all visual detail.
Next Question: What is saturation and how does it differ from contrast?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_18")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.ColorJitter(contrast=0.5)

print("=" * 55)
print("EXERCISE 18 — CONTRAST AUGMENTATION")
print("=" * 55)
print("ColorJitter(contrast=0.5)")
print("  contrast_factor ∈ [0.5, 1.5]")
print("  factor < 1 → low contrast (flat), factor > 1 → high contrast (vivid)")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 18 — Contrast Augmentation (4 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 5)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_18_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
