"""
Exercise 28 — Padding
======================
Data:        Random tensor (1, 3, 32, 32)
Program:     nn.Conv2d(3, 8, 3, padding=1)
Observation: Output is (1, 8, 32, 32) — spatial dimensions PRESERVED.
What Learned: padding=1 with kernel=3 pads 1 zero row/col on each side → output = input size.
Practical Use: Maintaining feature map size through deep networks (VGG, ResNet).
Limitation:  Zero-padding introduces artificial border values (zero context).
Next Question: What is stride and how does it downsample the feature map?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_28")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
x = torch.randn(1, 3, 32, 32)

conv_no_pad  = nn.Conv2d(3, 8, 3, padding=0)  # shrinks
conv_with_pad = nn.Conv2d(3, 8, 3, padding=1)  # preserves

y_no_pad   = conv_no_pad(x)
y_with_pad = conv_with_pad(x)

print("=" * 55)
print("EXERCISE 28 — PADDING")
print("=" * 55)
print(f"\nInput shape : {list(x.shape)}")
print(f"\nConv2d(3, 8, 3, padding=0) — no padding:")
print(f"  Output shape: {list(y_no_pad.shape)}")
print(f"  32 → 30  (shrinks by kernel_size-1=2)")
print(f"\nConv2d(3, 8, 3, padding=1) — 'same' padding:")
print(f"  Output shape: {list(y_with_pad.shape)}")
print(f"  32 → 32  (preserved!)")
print(f"\nFormula: output = (input + 2*padding - kernel) / stride + 1")
print(f"  With padding=1: (32 + 2 - 3) / 1 + 1 = 32  ✓")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Exercise 28 — Padding Effect on Feature Map Size", fontsize=12, fontweight="bold")

axes[0].imshow(x[0, 0].detach().numpy(), cmap="gray")
axes[0].set_title(f"Input\n{list(x.shape)}")
axes[0].axis("off")

axes[1].imshow(y_no_pad[0, 0].detach().numpy(), cmap="viridis")
axes[1].set_title(f"No padding (padding=0)\n{list(y_no_pad.shape)}\n32→30 (shrinks)")
axes[1].axis("off")

axes[2].imshow(y_with_pad[0, 0].detach().numpy(), cmap="viridis")
axes[2].set_title(f"'Same' padding (padding=1)\n{list(y_with_pad.shape)}\n32→32 (preserved)")
axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_28_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
