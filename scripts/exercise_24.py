"""
Exercise 24 — Combined Augmentation Pipeline
=============================================
Data:        Synthetic pet image
Program:     Combined: RandomResizedCrop + RandomHorizontalFlip + RandomRotation +
             ColorJitter + RandomPerspective
Observation: Each run of the pipeline produces a visually different image.
What Learned: Composition of augmentations greatly expands effective dataset size.
Practical Use: Standard ImageNet-style training pipeline for pet/animal classifiers.
Limitation:  Too many aggressive transforms can make training unstable.
Next Question: How do we decide WHICH augmentations are valid for a given task?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_24")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()

pipeline = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
])

print("=" * 55)
print("EXERCISE 24 — COMBINED AUGMENTATION PIPELINE")
print("=" * 55)
print("Pipeline:")
print("  1. RandomResizedCrop(224, scale=(0.7, 1.0))")
print("  2. RandomHorizontalFlip(p=0.5)")
print("  3. RandomRotation(degrees=15)")
print("  4. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)")
print("  5. RandomPerspective(distortion_scale=0.3, p=0.5)")
print("\n→ Each run produces a DIFFERENT result (stochastic pipeline)")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Exercise 24 — Combined Pipeline (8 runs — all different)", fontsize=13, fontweight="bold")

axes[0][0].imshow(img); axes[0][0].set_title("Original"); axes[0][0].axis("off")
idx = 1
for row in range(2):
    for col in range(4):
        if row == 0 and col == 0:
            continue
        torch.manual_seed(idx * 41)
        t = pipeline(img)
        axes[row][col].imshow(t)
        axes[row][col].set_title(f"Run {idx}")
        axes[row][col].axis("off")
        idx += 1

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_24_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
