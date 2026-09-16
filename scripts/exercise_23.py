"""
Exercise 23 — Random Erasing / Occlusion
==========================================
Data:        Synthetic pet image (as tensor)
Program:     transforms.ToTensor() + transforms.RandomErasing(p=1.0, scale=(0.1, 0.3))
Observation: A random rectangular region is replaced with zeros (black) or noise.
What Learned: Random erasing simulates real-world partial occlusion.
Practical Use: Makes model robust when part of the object is hidden (e.g., hand covers face).
Limitation:  If erasing covers the key discriminative region (eyes for face ID), accuracy drops.
Next Question: How can we combine all augmentations into a single pipeline?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_23")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()

# RandomErasing requires tensor input
to_tensor = T.ToTensor()
to_pil    = T.ToPILImage()
erasing   = T.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0)

print("=" * 55)
print("EXERCISE 23 — RANDOM ERASING / OCCLUSION")
print("=" * 55)
print("RandomErasing(p=1.0, scale=(0.1, 0.3), value=0)")
print("  scale → erased area is 10–30% of image")
print("  value=0 → erased region filled with zeros (black)")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 23 — Random Erasing / Occlusion (4 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 37)
    t = erasing(to_tensor(img))
    ax.imshow(to_pil(t)); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_23_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
