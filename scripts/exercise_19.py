"""
Exercise 19 — Saturation Augmentation
========================================
Data:        Synthetic pet image
Program:     transforms.ColorJitter(saturation=0.5)
Observation: Low saturation → greyscale; high saturation → oversaturated colours.
What Learned: Saturation controls colour vividness independent of brightness.
Practical Use: Simulates camera sensor differences, faded photos.
Limitation:  Saturation=0 removes all colour — becomes greyscale (label preserved for pets).
Next Question: Can we change hue (colour itself) without changing brightness?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_19")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.ColorJitter(saturation=0.5)

print("=" * 55)
print("EXERCISE 19 — SATURATION AUGMENTATION")
print("=" * 55)
print("ColorJitter(saturation=0.5)")
print("  saturation_factor ∈ [0.5, 1.5]")
print("  factor=0 → greyscale; factor=1 → original; factor=2 → oversaturated")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 19 — Saturation Augmentation (4 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 11)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_19_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
