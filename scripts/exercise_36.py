"""
Exercise 36 — CNN Architecture Thought Experiment
===================================================
No full training required — this exercise documents the evolution of CNN architectures.
Program:     Produces a clear architecture-evolution comparison visualisation.

Architectures covered:
  • LeNet  (1998)  — First practical CNN, convolutions on handwritten digits
  • AlexNet (2012) — GPU training, ReLU, Dropout, large-scale data (ImageNet)
  • VGG    (2014)  — Very deep with 3×3 convolutions only
  • Inception (2014) — Multiple kernel sizes in parallel, 1×1 bottleneck
  • ResNet  (2015) — Residual connections: y = F(x) + x, very deep networks

What Learned:
  Each architecture solved a specific engineering problem that the previous couldn't handle.

Practical Use:
  Choosing a backbone for transfer learning depends on accuracy/speed/memory tradeoff.

Limitation:
  All are eclipsed in accuracy by transformers (ViT) but remain important for embedded/edge.

Next Question:
  Can attention mechanisms (Transformers) be applied directly to image patches?
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_36")
os.makedirs(OUT, exist_ok=True)

print("=" * 65)
print("EXERCISE 36 — CNN ARCHITECTURE THOUGHT EXPERIMENT")
print("=" * 65)

architectures = {
    "LeNet\n(1998)": {
        "depth": 7,
        "params": "60K",
        "top5_err": "N/A",
        "core_idea": "First practical CNN.\nConvolution on images replaces hand-crafted features.\nTanh activation, avg-pool.",
        "problem_solved": "Digit recognition without hand-crafted feature engineering.",
        "colour": "#BBDEFB",
    },
    "AlexNet\n(2012)": {
        "depth": 8,
        "params": "60M",
        "top5_err": "16.4%",
        "core_idea": "GPU training (2 GPUs).\nReLU instead of tanh (no vanishing gradient).\nDropout for regularisation.\nLarge data (1.2M images).",
        "problem_solved": "Scale. Proved deep CNNs work on large datasets with GPUs.",
        "colour": "#C8E6C9",
    },
    "VGG\n(2014)": {
        "depth": 16,
        "params": "138M",
        "top5_err": "7.3%",
        "core_idea": "Stack MANY 3×3 convolutions.\nTwo 3×3 conv = one 5×5 but fewer parameters.\nVery deep = more representational power.",
        "problem_solved": "Showed depth matters. Simple recipe: only 3×3 conv.",
        "colour": "#F3E5F5",
    },
    "Inception\n(2014)": {
        "depth": 22,
        "params": "5M",
        "top5_err": "6.7%",
        "core_idea": "Run 1×1, 3×3, 5×5 convolutions IN PARALLEL.\n1×1 conv reduces channels before expensive 3×3/5×5 (bottleneck).\nConcat all outputs → rich multi-scale features.",
        "problem_solved": "Efficiency: more depth with fewer parameters via bottleneck.",
        "colour": "#FFF9C4",
    },
    "ResNet\n(2015)": {
        "depth": 152,
        "params": "25M",
        "top5_err": "3.6%",
        "core_idea": "Skip connections: y = F(x) + x\nGradients can flow directly through the shortcut.\nAllows training of 100+ layer networks without vanishing gradients.",
        "problem_solved": "Vanishing gradient in very deep networks.",
        "colour": "#FFE0B2",
    },
}

print("\n[Architecture Evolution Summary]")
print(f"\n{'Architecture':<12} {'Depth':>6} {'Params':>8}  Core Innovation")
print("-" * 70)
for arch, info in architectures.items():
    name = arch.replace("\n", " ")
    print(f"  {name:<12} {info['depth']:>6}  {info['params']:>8}  {info['core_idea'].split(chr(10))[0]}")

print("\n\n[ResNet Residual Connection Explained]")
print("  y = F(x) + x")
print("  x   = input to the block (before any convolution)")
print("  F(x)= what the block learned to add (residual)")
print("  y   = the final output")
print()
print("  Simple analogy: instead of learning the full answer,")
print("  the network only needs to learn the CORRECTION to the input.")
print("  If F(x)=0 (block learned nothing), y=x — the identity mapping.")
print("  This makes it easy for the network to be at least as good as")
print("  a shallower version of itself → no degradation with depth.")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Exercise 36 — CNN Architecture Evolution Timeline", fontsize=14, fontweight="bold")

# Top: timeline bar
ax_timeline = fig.add_axes([0.05, 0.72, 0.9, 0.18])
ax_timeline.axis("off")
years   = [1998, 2012, 2014, 2014, 2015]
x_pos   = [(y - 1995) / 23 for y in years]
colours = [info["colour"] for info in architectures.values()]
names   = list(architectures.keys())

ax_timeline.axhline(0.5, color="gray", linewidth=2, zorder=0)
for i, (xp, name, col, year) in enumerate(zip(x_pos, names, colours, years)):
    circ = mpatches.FancyBboxPatch((xp - 0.06, 0.2), 0.12, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor=col, edgecolor="gray", linewidth=1.5)
    ax_timeline.add_patch(circ)
    ax_timeline.text(xp, 0.5, name, ha="center", va="center", fontsize=8, fontweight="bold")
    ax_timeline.text(xp, 0.07, str(year), ha="center", va="center", fontsize=8, color="gray")
ax_timeline.set_xlim(0, 1); ax_timeline.set_ylim(0, 1)
ax_timeline.set_title("Timeline (1998 → 2015)", pad=3)

# Bottom: 5 info boxes
for i, (name, info) in enumerate(architectures.items()):
    ax = fig.add_axes([0.03 + i * 0.195, 0.03, 0.185, 0.65])
    ax.axis("off")
    ax.set_facecolor(info["colour"])
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                          boxstyle="round,pad=0.02",
                                          facecolor=info["colour"], edgecolor="#999",
                                          linewidth=1.5, clip_on=False))
    ax.text(0.5, 0.97, name, ha="center", va="top", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.88, f"Depth: {info['depth']} layers | Params: {info['params']}",
            ha="center", va="top", fontsize=7.5, color="#333", transform=ax.transAxes)
    ax.text(0.05, 0.78, "Core idea:", ha="left", va="top", fontsize=8, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.05, 0.70, info["core_idea"], ha="left", va="top", fontsize=7,
            transform=ax.transAxes, wrap=True,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2))
    ax.text(0.05, 0.30, "Problem solved:", ha="left", va="top", fontsize=8,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.22, info["problem_solved"], ha="left", va="top", fontsize=7,
            transform=ax.transAxes, wrap=True,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2))
    if "top5_err" in info and info["top5_err"] != "N/A":
        ax.text(0.5, 0.08, f"ImageNet Top-5 Error: {info['top5_err']}",
                ha="center", va="top", fontsize=7.5, color="#d32f2f",
                transform=ax.transAxes, fontweight="bold")

plt.savefig(os.path.join(OUT, "exercise_36_output.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── Residual block diagram ────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.axis("off")
fig2.suptitle("ResNet Residual Block: y = F(x) + x", fontsize=13, fontweight="bold")

boxes = [
    (0.1, 0.45, "x\n(input)", "#BBDEFB"),
    (0.35, 0.45, "Conv + BN\n+ ReLU", "#C8E6C9"),
    (0.60, 0.45, "Conv + BN", "#C8E6C9"),
    (0.82, 0.45, "⊕\n(add)", "#FFE0B2"),
    (0.95, 0.45, "y = F(x)+x\n(output)", "#BBDEFB"),
]
for (xb, yb, label, col) in boxes:
    rect = mpatches.FancyBboxPatch((xb - 0.07, yb - 0.12), 0.14, 0.24,
                                    boxstyle="round,pad=0.02", facecolor=col,
                                    edgecolor="gray", linewidth=1.5,
                                    transform=ax2.transAxes, clip_on=False)
    ax2.add_patch(rect)
    ax2.text(xb, yb, label, ha="center", va="center", fontsize=9,
             fontweight="bold", transform=ax2.transAxes)

# Arrows between boxes
for (x1, y1), (x2, y2) in zip(
        [(b[0]+0.07, b[1]) for b in boxes[:-1]],
        [(b[0]-0.07, b[1]) for b in boxes[1:]]):
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

# Skip connection arc
ax2.annotate("", xy=(0.75, 0.33), xytext=(0.17, 0.33),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", lw=2, color="tomato",
                             connectionstyle="arc3,rad=-0.4"))
ax2.text(0.46, 0.1, "Skip connection (identity shortcut x)", ha="center",
         fontsize=9, color="tomato", fontweight="bold", transform=ax2.transAxes)

plt.savefig(os.path.join(OUT, "exercise_36_residual_block.png"), dpi=120, bbox_inches="tight")
plt.close()

out_fig = os.path.join(OUT, "exercise_36_output.png")
print(f"\nFigure saved → {out_fig}")
print(f"Residual diagram saved → {OUT}/exercise_36_residual_block.png")
print("Done ✓")
