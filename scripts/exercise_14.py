"""
Exercise 14 — Translation (RandomAffine)
==========================================
Data:        Synthetic pet image
Program:     transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
Observation: Image shifts in random horizontal/vertical direction.
What Learned: Translation makes the model position-invariant.
Practical Use: Webcam feeds where subject is not always centred.
Limitation:  Large translations may move the subject out of frame.
Next Question: Can we combine translation with rotation in a single operation?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_14")
os.makedirs(OUT, exist_ok=True)

img = get_pet_image()

transform = T.RandomAffine(degrees=0, translate=(0.2, 0.2))

print("=" * 55)
print("EXERCISE 14 — TRANSLATION (RandomAffine)")
print("=" * 55)
print(f"\nOriginal size  : {img.size}")
print(f"Translate param: (0.2, 0.2)  → up to 20% of width/height in each direction")
print(f"Degrees        : 0  (no rotation — pure translation)")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 14 — Translation Augmentation (4 random samples)", fontsize=13, fontweight="bold")
axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
for i, ax in enumerate(axes[1:], 1):
    import torch; torch.manual_seed(i * 7)
    t = transform(img)
    ax.imshow(t); ax.set_title(f"Sample {i}"); ax.axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_14_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
