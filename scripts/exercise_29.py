"""
Exercise 29 — Stride
======================
Data:        Random tensor (1, 3, 32, 32)
Program:     nn.Conv2d(3, 8, 3, stride=2, padding=1)
Observation: Output is (1, 8, 16, 16) — spatial size halved.
What Learned: stride=2 skips every other position → 2x downsampling.
Practical Use: Reduces computation; replaces MaxPooling in some modern architectures.
Limitation:  Information is lost at boundaries — aliasing can occur.
Next Question: What is the ReLU activation and why is it essential after convolution?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_29")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
x = torch.randn(1, 3, 32, 32)

conv_s1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
conv_s2 = nn.Conv2d(3, 8, 3, stride=2, padding=1)

y_s1 = conv_s1(x)
y_s2 = conv_s2(x)

print("=" * 55)
print("EXERCISE 29 — STRIDE")
print("=" * 55)
print(f"\nInput shape  : {list(x.shape)}")
print(f"\nConv2d(3,8,3, stride=1, padding=1)  → {list(y_s1.shape)}")
print(f"Conv2d(3,8,3, stride=2, padding=1)  → {list(y_s2.shape)}")
print(f"\nFormula: output = floor((input + 2*padding - kernel) / stride) + 1")
print(f"  stride=1: floor((32 + 2 - 3) / 1) + 1 = 32  (no change)")
print(f"  stride=2: floor((32 + 2 - 3) / 2) + 1 = 16  (halved!)")
print(f"\nStride=2 is computationally cheaper than stride=1 followed by MaxPool.")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Exercise 29 — Stride Effect on Feature Map", fontsize=12, fontweight="bold")

axes[0].imshow(x[0, 0].detach().numpy(), cmap="gray")
axes[0].set_title(f"Input\n{list(x.shape)}")
axes[0].axis("off")

axes[1].imshow(y_s1[0, 0].detach().numpy(), cmap="viridis")
axes[1].set_title(f"stride=1\n{list(y_s1.shape)}\n(full resolution)")
axes[1].axis("off")

axes[2].imshow(y_s2[0, 0].detach().numpy(), cmap="viridis")
axes[2].set_title(f"stride=2\n{list(y_s2.shape)}\n(2× downsampled)")
axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_29_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
