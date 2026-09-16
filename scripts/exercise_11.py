"""
Exercise 11 — Horizontal Flip
==============================
Data:        Synthetic pet image (224x224 RGB)
Program:     torchvision.transforms.functional.hflip()
Observation: Image is mirrored left-right. Cat eyes swap positions.
What Learned: Horizontal flip is a label-preserving augmentation for most natural images.
Practical Use: Doubles training data for symmetric scenes (animals, faces, objects).
Limitation:  Not valid for text, asymmetric logos, or directional signs.
Next Question: What if the scene has a left-right asymmetry that matters (e.g., driving direction)?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_11")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
flipped = F.hflip(img)

print("=" * 55)
print("EXERCISE 11 — HORIZONTAL FLIP")
print("=" * 55)
print(f"\nOriginal size : {img.size}")
print(f"Flipped size  : {flipped.size}")
print("Pixel check — top-left pixel of original is now top-right pixel of flipped.")
import numpy as np
orig_arr = np.array(img)
flip_arr = np.array(flipped)
print(f"  Original [0,0] pixel  : {orig_arr[0,0]}")
print(f"  Flipped [0,-1] pixel  : {flip_arr[0,-1]}  (should match)")
print(f"  Match: {np.allclose(orig_arr[0,0], flip_arr[0,-1])}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Exercise 11 — Horizontal Flip (F.hflip)", fontsize=13, fontweight="bold")
axes[0].imshow(img);     axes[0].set_title("Original");        axes[0].axis("off")
axes[1].imshow(flipped); axes[1].set_title("Horizontal Flip"); axes[1].axis("off")
plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_11_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
