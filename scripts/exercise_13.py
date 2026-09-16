"""
Exercise 13 — Rotation (~20°)
==============================
Data:        Synthetic pet image
Program:     torchvision.transforms.functional.rotate(img, 20)
Observation: Image is rotated ~20° counterclockwise; corners fill with black.
What Learned: Rotation augmentation makes the model pose-invariant.
Practical Use: Aerial imagery, medical imaging, pet recognition from odd angles.
Limitation:  Large rotations (>30°) may look unnatural; corners filled with zero (black).
Next Question: Can we also change scale or position at the same time?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_13")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
rotated = F.rotate(img, angle=20)

print("=" * 55)
print("EXERCISE 13 — ROTATION (~20°)")
print("=" * 55)
print(f"\nOriginal size  : {img.size}")
print(f"Rotated size   : {rotated.size}  (same — image padded with zeros)")
print(f"Rotation angle : 20° (counterclockwise)")
print("\nNote: Corners contain black (zero) fill — this is fill_value=0 default.")
print("The label 'cat' / 'dog' is preserved — a rotated animal is still that animal.")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Exercise 13 — Rotation Augmentation", fontsize=13, fontweight="bold")
axes[0].imshow(img);            axes[0].set_title("Original");  axes[0].axis("off")
axes[1].imshow(rotated);        axes[1].set_title("+20° rotation"); axes[1].axis("off")
axes[2].imshow(F.rotate(img, angle=-20)); axes[2].set_title("-20° rotation"); axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_13_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
