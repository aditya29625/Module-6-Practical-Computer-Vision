"""
Exercise 27 — Pass Data Through Convolution
=============================================
Data:        Random tensor x = torch.randn(1, 3, 32, 32)
Program:     y = conv(x); print input/output shapes
Observation: Input (1,3,32,32) → Output (1,8,30,30)
What Learned: Without padding, each spatial dimension shrinks by (kernel_size - 1) = 2.
Practical Use: Understanding shape flow is critical for designing CNN architectures.
Limitation:  Without padding, repeated convolutions shrink the feature map to nothing.
Next Question: How does padding prevent this spatial shrinkage?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_27")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
x    = torch.randn(1, 3, 32, 32)
conv = nn.Conv2d(3, 8, 3)
y    = conv(x)

print("=" * 55)
print("EXERCISE 27 — PASS DATA THROUGH CONVOLUTION")
print("=" * 55)
print(f"\nInput  x : shape = {list(x.shape)}")
print(f"  [1, 3, 32, 32] → batch=1, channels=3, H=32, W=32")
print(f"\nConv2d(3, 8, 3) applied ...")
print(f"\nOutput y : shape = {list(y.shape)}")
print(f"  [1, 8, 30, 30] → batch=1, out_channels=8, H=30, W=30")
print(f"\nWhy does 32 → 30?")
print(f"  Output size = (Input - kernel + 2*padding) / stride + 1")
print(f"             = (32 - 3 + 0) / 1 + 1 = 30")
print(f"  The kernel (3×3) cannot be centred on the border pixels → loses 1 pixel each side")
print(f"\nOutput statistics:")
print(f"  min={y.data.min():.4f}  max={y.data.max():.4f}  mean={y.data.mean():.4f}")

# Visualise output feature maps
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle("Exercise 27 — Output Feature Maps after Conv2d(3,8,3)", fontsize=12, fontweight="bold")

axes[0][0].imshow(x[0].permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")
axes[0][0].set_title("Input (ch 0 of 3)")
axes[0][0].axis("off")

for i in range(8):
    row, col = (i // 5), (i % 5) + (1 if i < 5 else 0)
    if i < 5:
        ax = axes[0][i+1] if i < 4 else axes[1][0]
    else:
        ax = axes[1][i-4]
    fm = y[0, i].detach().numpy()
    ax.imshow(fm, cmap="viridis")
    ax.set_title(f"Feature map {i+1}\nshape: {list(fm.shape)}")
    ax.axis("off")

axes[1][4].axis("off")
plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_27_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
