"""
Exercise 31 — Max Pooling
===========================
Data:        Manually constructed 4×4 tensor
Program:     nn.MaxPool2d(kernel_size=2)
Observation: 4×4 → 2×2 output: each 2×2 block reduced to its maximum value.
What Learned: MaxPool retains the STRONGEST activation in each region → translation invariance.
Practical Use: Dimensionality reduction; makes features invariant to small shifts.
Limitation:  Exact position of the maximum is lost — not suitable for tasks needing precise location.
Next Question: How do these building blocks combine into a complete CNN?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_31")
os.makedirs(OUT, exist_ok=True)

# Manually crafted 4x4 tensor
x = torch.tensor([[
    [[ 1., 3.,  2., 4.],
     [ 5., 6.,  7., 8.],
     [ 9., 2.,  1., 3.],
     [ 4., 7.,  8., 6.]]
]])

pool = nn.MaxPool2d(kernel_size=2)
y = pool(x)

print("=" * 55)
print("EXERCISE 31 — MAX POOLING")
print("=" * 55)
print(f"\nInput tensor (1,1,4,4):")
print(x[0,0].numpy())
print(f"\nMaxPool2d(kernel_size=2) applied:")
print(f"Output tensor (1,1,2,2):")
print(y[0,0].detach().numpy())
print(f"\nExplanation (2×2 blocks):")
print(f"  Top-left    [1,3,5,6]  → max = 6")
print(f"  Top-right   [2,4,7,8]  → max = 8")
print(f"  Bottom-left [9,2,4,7]  → max = 9")
print(f"  Bottom-right[1,3,8,6]  → max = 8")
print(f"\nOutput shape: {list(x.shape)} → {list(y.shape)}")
print(f"  (4×4) → (2×2): spatial dimensions halved")
print(f"\nWhat is RETAINED: the strongest activation (presence of feature)")
print(f"What is LOST:     exact position within each 2×2 region")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Exercise 31 — Max Pooling (4×4 → 2×2)", fontsize=13, fontweight="bold")

x_np = x[0,0].numpy()
y_np = y[0,0].detach().numpy()

im0 = axes[0].imshow(x_np, cmap="YlOrRd", vmin=1, vmax=9)
axes[0].set_title(f"Input (4×4)\n{list(x.shape)}")
for r in range(4):
    for c in range(4):
        axes[0].text(c, r, str(int(x_np[r,c])), ha="center", va="center",
                     fontsize=14, fontweight="bold")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(y_np, cmap="YlOrRd", vmin=1, vmax=9)
axes[1].set_title(f"Output (2×2)\n{list(y.shape)}\nMaxPool2d(2)")
for r in range(2):
    for c in range(2):
        axes[1].text(c, r, str(int(y_np[r,c])), ha="center", va="center",
                     fontsize=16, fontweight="bold")
plt.colorbar(im1, ax=axes[1])

# Visualise blocks
x_big = np.kron(x_np, np.ones((60, 60)))
axes[2].imshow(x_big, cmap="YlOrRd", vmin=1, vmax=9)
axes[2].set_title("2×2 pooling regions highlighted")
for r in [0, 120]:
    axes[2].axhline(r - 0.5, color='blue', linewidth=2)
    axes[2].axvline(r - 0.5, color='blue', linewidth=2)
for r in range(4):
    for c in range(4):
        color = "blue" if (r < 2 and c < 2) or (r >= 2 and c >= 2) else "red"
        axes[2].text(c*60+30, r*60+30, str(int(x_np[r,c])),
                     ha="center", va="center", fontsize=14, color=color, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_31_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
