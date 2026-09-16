"""
Exercise 35 — Confusion Matrix
================================
Data:        y_true = [1,1,1,0,0,0]  y_pred = [1,0,1,0,0,1]
Program:     sklearn.metrics.confusion_matrix + visualisation
Observation: CM shows 2 TP, 1 FN, 1 FP, 2 TN — model has one false alarm and one miss.
What Learned: Confusion matrix shows all four outcomes simultaneously.
Practical Use: Evaluating binary or multi-class classifiers beyond a single number.
Limitation:  For many classes the matrix becomes large and hard to read without normalisation.
Next Question: How do we combine precision and recall into a single metric (F1)?
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_35")
os.makedirs(OUT, exist_ok=True)

y_true = [1, 1, 1, 0, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=" * 55)
print("EXERCISE 35 — CONFUSION MATRIX")
print("=" * 55)
print(f"\ny_true : {y_true}")
print(f"y_pred : {y_pred}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nBreakdown:")
print(f"  True Negatives  (TN): {tn}  — predicted 0, actually 0")
print(f"  False Positives (FP): {fp}  — predicted 1, actually 0  (false alarm)")
print(f"  False Negatives (FN): {fn}  — predicted 0, actually 1  (missed positive)")
print(f"  True Positives  (TP): {tp}  — predicted 1, actually 1")
print(f"\nDerived metrics:")
print(f"  Accuracy  : (TP+TN)/(TP+TN+FP+FN) = {(tp+tn)/len(y_true):.2f}")
print(f"  Precision : TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.2f}  (of predicted positives, how many are right?)")
print(f"  Recall    : TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.2f}  (of actual positives, how many did we catch?)")
print(f"\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["Class 0 (Neg)", "Class 1 (Pos)"]))

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Exercise 35 — Confusion Matrix Visualisation", fontsize=13, fontweight="bold")

# Heatmap
labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                   [f"FN\n{fn}", f"TP\n{tp}"]])
colours_cm = np.array([[tn, fp], [fn, tp]])
im = axes[0].imshow(colours_cm, cmap="Blues", vmin=0, vmax=3)
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Predicted\nNegative (0)", "Predicted\nPositive (1)"])
axes[0].set_yticklabels(["Actual\nNegative (0)", "Actual\nPositive (1)"])
for r in range(2):
    for c in range(2):
        axes[0].text(c, r, labels[r, c], ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if colours_cm[r, c] > 1.5 else "black")
plt.colorbar(im, ax=axes[0])
axes[0].set_title("Confusion Matrix\n(annotated with TP/FP/TN/FN)")

# Scatter of individual predictions
colours_pred = []
labels_pred  = []
for yt, yp in zip(y_true, y_pred):
    if yt == 1 and yp == 1:   colours_pred.append("green");  labels_pred.append("TP")
    elif yt == 0 and yp == 0: colours_pred.append("blue");   labels_pred.append("TN")
    elif yt == 0 and yp == 1: colours_pred.append("orange"); labels_pred.append("FP")
    else:                      colours_pred.append("red");    labels_pred.append("FN")

jitter = np.random.default_rng(0).normal(0, 0.05, len(y_true))
axes[1].scatter(np.array(y_true) + jitter, np.array(y_pred) + jitter*0.5,
                c=colours_pred, s=200, zorder=5, edgecolors="black", linewidths=1.5)
for i, (yt, yp, lbl) in enumerate(zip(y_true, y_pred, labels_pred)):
    axes[1].text(yt + jitter[i] + 0.05, yp + jitter[i]*0.5 + 0.04, lbl, fontsize=11, fontweight="bold")

axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Actual 0", "Actual 1"])
axes[1].set_yticklabels(["Predicted 0", "Predicted 1"])
axes[1].set_xlim(-0.5, 1.5); axes[1].set_ylim(-0.5, 1.5)
axes[1].set_title("Individual predictions\n(per sample)")
axes[1].grid(True, alpha=0.3)

legend_handles = [
    mpatches.Patch(color="green",  label="TP — correct positive"),
    mpatches.Patch(color="blue",   label="TN — correct negative"),
    mpatches.Patch(color="orange", label="FP — false alarm"),
    mpatches.Patch(color="red",    label="FN — missed positive"),
]
axes[1].legend(handles=legend_handles, loc="upper left", fontsize=9)

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_35_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
