"""
Exercise 21 — Blur as Augmentation (GaussianBlur)
===================================================
Data:        Synthetic pet image
Program:     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
Observation: Image appears softly blurred; high-frequency details fade.
What Learned: Blur augmentation simulates out-of-focus or motion blur.
Practical Use: Makes models robust to defocus, fast-moving subjects.
Limitation:  Extreme blur removes all detail — classification becomes impossible.
Next Question: What if parts of the image are erased entirely (occluded)?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_21")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()
transform = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

print("=" * 55)
print("EXERCISE 21 — BLUR AS AUGMENTATION")
print("=" * 55)
print("GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))")
print("  sigma sampled uniformly from [0.1, 2.0]")
print("  Low sigma → slight blur; high sigma → significant softening")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Exercise 21 — GaussianBlur Augmentation", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    torch.manual_seed(i * 23)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_21_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
