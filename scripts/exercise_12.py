"""
Exercise 12 — Vertical Flip
==============================
Data:        Synthetic pet image
Program:     torchvision.transforms.functional.vflip()
Observation: Image is mirrored top-to-bottom — pet appears upside-down.
What Learned: Vertical flip preserves pixel information but changes semantic meaning.
Practical Use: Useful for microscopy, satellite imagery (no canonical "up").
Limitation:  Rarely valid for natural scenes — upside-down pets/people look unnatural.
Next Question: What rotation angles are safe augmentations for pet recognition?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_12")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
flipped_v = F.vflip(img)

print("=" * 55)
print("EXERCISE 12 — VERTICAL FLIP")
print("=" * 55)
print(f"\nOriginal size  : {img.size}")
print(f"Flipped size   : {flipped_v.size}")
print("The image is now upside-down.")
print("For satellite or microscopy images this would be acceptable.")
print("For pet classification this could confuse the model (rare in real data).")

import numpy as np
orig_arr = np.array(img)
flip_arr = np.array(flipped_v)
print(f"\n  Original top-left  [0,0] pixel : {orig_arr[0,0]}")
print(f"  Flipped bottom-left [-1,0] pixel: {flip_arr[-1,0]}  (should match)")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Exercise 12 — Vertical Flip (F.vflip)", fontsize=13, fontweight="bold")
axes[0].imshow(img);      axes[0].set_title("Original");      axes[0].axis("off")
axes[1].imshow(flipped_v); axes[1].set_title("Vertical Flip"); axes[1].axis("off")
plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_12_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
