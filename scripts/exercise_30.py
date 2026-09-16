"""
Exercise 30 — ReLU Activation
================================
Data:        Small 1D tensor with negative and positive values
Program:     nn.ReLU applied to [-5, -2, 0, 2, 8]
Observation: Negative values → 0; positive values → unchanged.
What Learned: ReLU(x) = max(0, x). It introduces non-linearity without vanishing gradient.
Practical Use: Standard activation in virtually all modern CNNs.
Limitation:  "Dying ReLU" — neurons stuck at 0 if inputs are always negative.
Next Question: What is MaxPooling and what spatial information does it discard?
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_30")
os.makedirs(OUT, exist_ok=True)

relu = nn.ReLU()
x = torch.tensor([-5., -2., 0., 2., 8.])
y = relu(x)

print("=" * 55)
print("EXERCISE 30 — ReLU ACTIVATION")
print("=" * 55)
print(f"\nInput  : {x.tolist()}")
print(f"Output : {y.tolist()}")
print(f"\nReLU(x) = max(0, x)")
print(f"  -5 → 0   (clamped)")
print(f"  -2 → 0   (clamped)")
print(f"   0 → 0   (unchanged)")
print(f"   2 → 2   (unchanged)")
print(f"   8 → 8   (unchanged)")
print(f"\nWhy ReLU matters:")
print(f"  1. Non-linearity: stacking linear layers alone cannot learn complex functions")
print(f"  2. Fast computation: just a max(0, x) — no exp() like sigmoid/tanh")
print(f"  3. No vanishing gradient for positive values (gradient = 1)")

# Wider demo
x_dense = torch.linspace(-6, 6, 200)
y_dense = relu(x_dense)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Exercise 30 — ReLU Activation Function", fontsize=13, fontweight="bold")

# Plot ReLU function
axes[0].plot(x_dense.numpy(), x_dense.numpy(), 'b--', alpha=0.4, label="y = x (identity)")
axes[0].plot(x_dense.numpy(), y_dense.detach().numpy(), 'r-', linewidth=2.5, label="ReLU(x)")
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(0, color='k', linewidth=0.5)
axes[0].set_xlabel("Input x"); axes[0].set_ylabel("Output ReLU(x)")
axes[0].set_title("ReLU function: max(0, x)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Bar chart of the example
colours = ['red' if v < 0 else 'green' for v in x.tolist()]
bars = axes[1].bar(range(5), x.tolist(), color=colours, alpha=0.5, label="Input")
axes[1].bar(range(5), y.tolist(), color=['steelblue']*5, alpha=0.8, label="ReLU output")
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_xticks(range(5)); axes[1].set_xticklabels([f"{v:.0f}" for v in x.tolist()])
axes[1].set_title("Example: relu([-5,-2,0,2,8])")
axes[1].set_xlabel("Input value"); axes[1].set_ylabel("Value"); axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_30_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
