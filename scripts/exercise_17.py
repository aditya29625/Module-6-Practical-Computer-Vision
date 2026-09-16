"""
Exercise 17 — Brightness Augmentation
=======================================
Data:        Synthetic pet image
Program:     transforms.ColorJitter(brightness=0.5)
Observation: Image becomes randomly brighter or darker.
What Learned: Brightness augmentation simulates different lighting conditions.
Practical Use: Outdoor CV systems (dawn, dusk, overcast, sunny).
Limitation:  Extreme brightness can saturate or completely darken the image.
Next Question: What other colour properties (contrast, saturation, hue) can we jitter?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_17")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.ColorJitter(brightness=0.5)

print("=" * 55)
print("EXERCISE 17 — BRIGHTNESS AUGMENTATION")
print("=" * 55)
print(f"\nColorJitter(brightness=0.5)")
print(f"  → brightness_factor sampled from [1-0.5, 1+0.5] = [0.5, 1.5]")
print(f"  → factor < 1 darkens, factor > 1 brightens")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 17 — Brightness Augmentation (4 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 3)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_17_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
