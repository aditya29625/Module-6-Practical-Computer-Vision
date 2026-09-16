"""
Exercise 26 — Create a Learnable Convolution
=============================================
Data:        No image required — we inspect the weight tensor directly.
Program:     nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
Observation: conv.weight.shape = [8, 3, 3, 3]
What Learned: [out_channels, in_channels, kH, kW] — each filter is learned during training.
Practical Use: First layer of any CNN takes raw RGB and learns edge/colour detectors.
Limitation:  Random initialisation produces random outputs until training adjusts weights.
Next Question: What happens to the output tensor shape when data is passed through?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_26")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3
)

print("=" * 55)
print("EXERCISE 26 — LEARNABLE CONVOLUTION (nn.Conv2d)")
print("=" * 55)
print(f"\nconv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)")
print(f"\nWeight shape : {conv.weight.shape}")
print(f"  [8, 3, 3, 3] means:")
print(f"  • 8  → out_channels (8 different learned filters)")
print(f"  • 3  → in_channels  (each filter has depth 3 for R, G, B)")
print(f"  • 3  → kernel height (3 pixels tall)")
print(f"  • 3  → kernel width  (3 pixels wide)")
print(f"\nBias shape   : {conv.bias.shape}  (one bias per output channel)")
total_params = sum(p.numel() for p in conv.parameters())
print(f"Total params : {total_params}  (8×3×3×3 weights + 8 biases)")
print(f"\nWeight stats (random init):")
print(f"  min={conv.weight.data.min():.4f}  max={conv.weight.data.max():.4f}  mean={conv.weight.data.mean():.4f}")

# Visualise all 8 filters (first channel only for clarity)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Exercise 26 — 8 Learned Convolution Filters (channel 0 shown)", fontsize=12, fontweight="bold")
for idx, ax in enumerate(axes.flatten()):
    kernel = conv.weight.data[idx, 0].numpy()  # shape [3,3]
    im = ax.imshow(kernel, cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_title(f"Filter {idx+1}")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{kernel[r,c]:.2f}", ha="center", va="center", fontsize=7)
    ax.axis("off")

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_26_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
