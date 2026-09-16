"""
Exercise 20 — Hue Augmentation
================================
Data:        Synthetic pet image
Program:     transforms.ColorJitter(hue=0.1)
Observation: Subtle hue shifts change the colour tone (orange → reddish or yellowish).
What Learned: Hue shifts the entire colour wheel by a fraction; 0.1 = ±36°.
Practical Use: Makes models colour-agnostic for objects where colour is not discriminative.
Limitation:  For tasks where colour IS the label (traffic lights, fruit ripeness), hue augmentation
             can DESTROY the label — use with caution.
Next Question: Should hue augmentation be used for traffic-light or fruit-colour classification?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_20")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.ColorJitter(hue=0.1)

print("=" * 55)
print("EXERCISE 20 — HUE AUGMENTATION")
print("=" * 55)
print("ColorJitter(hue=0.1)")
print("  hue_factor ∈ [-0.1, 0.1]  (fraction of 360°)")
print("  This shifts all pixel hues by up to ±36°")
print("\n⚠ WARNING: Hue augmentation can DESTROY labels for colour-dependent tasks!")
print("  Example: a RED traffic light shifted to GREEN changes the label.")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 20 — Hue Augmentation (4 samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 19)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_20_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
