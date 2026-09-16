"""
Exercise 32 — Build a Tiny CNN
================================
Data:        Random tensor (1, 3, 32, 32) — CIFAR-10 size
Program:     Sequential CNN with 2 conv blocks + flatten + linear
Observation: Model summary shows progressive downsampling and channel expansion.
What Learned: CNNs alternate between feature extraction (Conv+ReLU) and downsampling (Pool).
Practical Use: Template for a CIFAR-10 classifier — 10-class output.
Limitation:  No batch normalization, no dropout — will overfit on small datasets.
Next Question: What does the output vector mean for classification?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_32")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),   # → (1,16,32,32)
    nn.ReLU(),
    nn.MaxPool2d(2),                   # → (1,16,16,16)
    nn.Conv2d(16, 32, 3, padding=1),  # → (1,32,16,16)
    nn.ReLU(),
    nn.MaxPool2d(2),                   # → (1,32,8,8)
    nn.Flatten(),                      # → (1,2048)
    nn.Linear(32 * 8 * 8, 10)         # → (1,10)
)

print("=" * 55)
print("EXERCISE 32 — TINY CNN ARCHITECTURE")
print("=" * 55)
print(f"\nModel:\n{model}")

# Trace tensor shape through every layer
x = torch.randn(1, 3, 32, 32)
print(f"\n{'Layer':<30} {'Output Shape'}")
print("-" * 55)
print(f"{'Input':<30} {list(x.shape)}")

hooks_outputs = []
def make_hook(name):
    def hook(module, inp, out):
        hooks_outputs.append((name, list(out.shape)))
    return hook

layer_names = [
    "Conv2d(3,16,3,pad=1)", "ReLU", "MaxPool2d(2)",
    "Conv2d(16,32,3,pad=1)", "ReLU", "MaxPool2d(2)",
    "Flatten", "Linear(2048,10)"
]
handles = []
for name, layer in zip(layer_names, model.children()):
    handles.append(layer.register_forward_hook(make_hook(name)))

with torch.no_grad():
    _ = model(x)

for h in handles:
    h.remove()

for name, shape in hooks_outputs:
    print(f"  {name:<28} {shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")

# Architecture diagram as matplotlib figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
fig.suptitle("Exercise 32 — Tiny CNN Architecture & Shape Trace", fontsize=13, fontweight="bold")

layers_info = [
    ("Input\n(1,3,32,32)", "#BBDEFB"),
    ("Conv2d\n(1,16,32,32)", "#C8E6C9"),
    ("ReLU\n(1,16,32,32)", "#F3E5F5"),
    ("MaxPool\n(1,16,16,16)", "#FFE0B2"),
    ("Conv2d\n(1,32,16,16)", "#C8E6C9"),
    ("ReLU\n(1,32,16,16)", "#F3E5F5"),
    ("MaxPool\n(1,32,8,8)", "#FFE0B2"),
    ("Flatten\n(1,2048)", "#B2EBF2"),
    ("Linear\n(1,10)", "#FFCDD2"),
]

n = len(layers_info)
x_pos = [i * (14 / n) + 0.5 for i in range(n)]
for i, (label, colour) in enumerate(layers_info):
    rect = plt.Rectangle((x_pos[i]-0.55, 0.2), 1.1, 0.6,
                          linewidth=1.5, edgecolor="#555", facecolor=colour)
    ax.add_patch(rect)
    ax.text(x_pos[i], 0.5, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
    if i < n - 1:
        ax.annotate("", xy=(x_pos[i+1]-0.55, 0.5), xytext=(x_pos[i]+0.55, 0.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.set_xlim(0, 14); ax.set_ylim(0, 1)

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_32_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
