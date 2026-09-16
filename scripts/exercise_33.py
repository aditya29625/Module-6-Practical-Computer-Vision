"""
Exercise 33 — Classification Output (Logits → Probabilities → Predicted Class)
===============================================================================
Data:        Random input tensor (1, 3, 32, 32)
Program:     Pass through model from Ex32 → logits → softmax → argmax
Observation: Raw logits are unconstrained; softmax converts to valid probability distribution.
What Learned: Cross-entropy loss works on logits; softmax gives interpretable probabilities.
Practical Use: Final classification head of any image classifier.
Limitation:  Softmax max-probability is not calibrated confidence (can be overconfident).
Next Question: How do we evaluate whether a classifier is actually accurate and fair?
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_33")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
)
model.eval()

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

x = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    output = model(x)

logits = output[0]
probs  = torch.softmax(logits, dim=0)
pred   = torch.argmax(probs).item()

print("=" * 55)
print("EXERCISE 33 — CLASSIFICATION OUTPUT")
print("=" * 55)
print(f"\nInput shape  : {list(x.shape)}")
print(f"Output shape : {list(output.shape)}")
print(f"\nRaw logits (not probabilities):")
for i, (cls, l) in enumerate(zip(CIFAR10_CLASSES, logits.tolist())):
    print(f"  [{i}] {cls:<12}: {l:+.4f}")
print(f"\nWhy raw logits are NOT probabilities:")
print(f"  • Logits can be negative (impossible for a probability)")
print(f"  • Logits don't sum to 1")
print(f"  • Example: sum of logits = {logits.sum():.4f}  (not 1!)")
print(f"\nSoftmax probabilities (sum to 1):")
for i, (cls, p) in enumerate(zip(CIFAR10_CLASSES, probs.tolist())):
    bar = "█" * int(p * 50)
    print(f"  [{i}] {cls:<12}: {p:.4f}  {bar}")
print(f"\nSum of probabilities: {probs.sum():.6f}  ✓")
print(f"\nPredicted class: [{pred}] {CIFAR10_CLASSES[pred]}")
print(f"Confidence     : {probs[pred].item()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Exercise 33 — Logits vs Softmax Probabilities", fontsize=13, fontweight="bold")

x_ticks = range(10)
axes[0].bar(x_ticks, logits.numpy(), color="steelblue", edgecolor="black")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1, label="zero line")
axes[0].set_xticks(x_ticks); axes[0].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
axes[0].set_title("Raw Logits\n(can be negative, don't sum to 1)")
axes[0].set_ylabel("Logit value"); axes[0].legend()

colours = ["tomato" if i == pred else "steelblue" for i in range(10)]
axes[1].bar(x_ticks, probs.numpy(), color=colours, edgecolor="black")
axes[1].set_xticks(x_ticks); axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
axes[1].set_title(f"Softmax Probabilities\n(sum=1, predicted: '{CIFAR10_CLASSES[pred]}')")
axes[1].set_ylabel("Probability (0–1)")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_33_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
