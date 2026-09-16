"""
Exercise 34 — Accuracy Can Mislead
=====================================
Data:        Simulated manufacturing inspection scenario (950 normal, 50 defective)
Program:     Model that always predicts NORMAL — calculate all metrics
Observation: 95% accuracy, but 0% recall for the defect class.
What Learned: Accuracy is misleading on imbalanced datasets. Precision/Recall matter more.
Practical Use: Quality control, medical diagnosis, fraud detection.
Limitation:  Even precision/recall can be gamed — use F1 or domain-specific cost weighting.
Next Question: How do we visualise all TP/FP/TN/FN values simultaneously?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_34")
os.makedirs(OUT, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────
total        = 1000
n_normal     = 950
n_defective  = 50
# Model predicts NORMAL for every image
tp = 0    # correctly predicted defective → but model never predicts defective
fp = 0    # normal predicted as defective → model never predicts defective
tn = n_normal   # correctly predicted normal
fn = n_defective # defective predicted as normal (ALL of them)

accuracy  = (tp + tn) / total
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("=" * 65)
print("EXERCISE 34 — ACCURACY CAN MISLEAD")
print("=" * 65)
print(f"\nDataset: {total} images")
print(f"  Normal    : {n_normal}  ({n_normal/total*100:.0f}%)")
print(f"  Defective : {n_defective}  ({n_defective/total*100:.0f}%)")
print(f"\nModel strategy: ALWAYS predict NORMAL (worst possible defect detector)")
print(f"\n{'Metric':<20} {'Value'}")
print("-" * 35)
print(f"  {'True Positives':<18} {tp}   (defects correctly caught)")
print(f"  {'False Positives':<18} {fp}   (normal wrongly flagged as defect)")
print(f"  {'True Negatives':<18} {tn}  (normal correctly classified)")
print(f"  {'False Negatives':<18} {fn}  (MISSED defects — very dangerous!)")
print(f"\n  {'Accuracy':<18} {accuracy*100:.1f}%  ← looks great!")
print(f"  {'Recall':<18} {recall*100:.1f}%   ← catastrophically bad")
print(f"  {'Precision':<18} {precision*100:.1f}%   (undefined → 0 by convention)")
print(f"  {'F1 Score':<18} {f1*100:.1f}%   ← reflects true failure")
print(f"\n⚠ 95% accuracy but ZERO defects caught. This system is useless for QC.")
print(f"  Every defective product ships. Recall=0 means the detector does not work.")

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Exercise 34 — Why Accuracy Can Mislead (Imbalanced Dataset)", fontsize=13, fontweight="bold")

# Pie chart of class distribution
axes[0].pie([n_normal, n_defective], labels=["Normal\n(950)", "Defective\n(50)"],
            colors=["#90CAF9", "#EF9A9A"], autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11})
axes[0].set_title("Class distribution\n(heavily imbalanced)")

# Confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(["Pred: Normal", "Pred: Defect"])
axes[1].set_yticklabels(["True: Normal", "True: Defect"])
for r in range(2):
    for c in range(2):
        axes[1].text(c, r, str(cm[r,c]), ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if cm[r,c] > 400 else "black")
axes[1].set_title("Confusion Matrix\n(model always says Normal)")
plt.colorbar(im, ax=axes[1])

# Metrics bar chart
metrics = ["Accuracy\n95%", "Recall\n0%", "Precision\nN/A→0%", "F1\n0%"]
values  = [accuracy, recall, precision, f1]
bar_colours = ["#66BB6A", "#EF5350", "#EF5350", "#EF5350"]
axes[2].bar(range(4), [v * 100 for v in values], color=bar_colours, edgecolor="black", width=0.6)
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(metrics)
axes[2].set_ylim(0, 110); axes[2].set_ylabel("Score (%)")
axes[2].set_title("Metrics Summary\n(accuracy misleads; recall=0 is critical)")
for i, v in enumerate(values):
    axes[2].text(i, v * 100 + 2, f"{v*100:.0f}%", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_34_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
