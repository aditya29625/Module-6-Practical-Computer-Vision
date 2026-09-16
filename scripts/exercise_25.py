"""
Exercise 25 — Augmentation Policy Design
==========================================
Data:        Oxford-IIIT Pet dataset (represented by synthetic pet image)
Task:        Pet breed classification (37 classes, cats and dogs)
Program:     Evaluate each augmentation as YES / NO / MAYBE with justification.
Observation: Not all augmentations preserve the semantic label "cat breed X".
What Learned: Augmentation policy design requires domain knowledge about label sensitivity.
Practical Use: Production training pipelines for any classification task.
Limitation:  Wrong augmentations can silently degrade model performance.
Next Question: How do CNNs actually learn these visual features?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pet_helper import get_pet_image
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_25")
os.makedirs(OUT, exist_ok=True)

# ── Augmentation Policy Table ─────────────────────────────────────────────────
policy = [
    # (Augmentation, Decision, Reason)
    ("Horizontal Flip",         "YES",   "Pets look the same facing left or right"),
    ("Vertical Flip",           "NO",    "Upside-down pets are not realistic — breaks label"),
    ("Rotation ±15°",           "YES",   "Small tilts are realistic; breed still recognisable"),
    ("Rotation ±90°",           "MAYBE", "Cats sometimes lie sideways; risky for > 30°"),
    ("Translation (20%)",       "YES",   "Pet not always centred; position-invariance needed"),
    ("Scale / Zoom",            "YES",   "Pets appear at varying distances from camera"),
    ("Random Crop",             "YES",   "Forces local feature learning; standard practice"),
    ("Brightness Jitter",       "YES",   "Simulates different lighting conditions"),
    ("Contrast Jitter",         "YES",   "Simulates foggy / crisp environments"),
    ("Saturation Jitter",       "YES",   "Simulates camera sensor variation"),
    ("Hue Jitter",              "MAYBE", "Small hue OK for breed ID, but risky if coat colour matters"),
    ("Gaussian Blur",           "YES",   "Simulates out-of-focus or low-res cameras"),
    ("Perspective Distortion",  "MAYBE", "Mild only — extreme distortion changes appearance significantly"),
    ("Random Erasing",          "YES",   "Simulates partial occlusion; model must be robust"),
    ("Grayscale",               "MAYBE", "OK if breed distinguishable by shape; NO if coat colour is key"),
    ("Shear",                   "MAYBE", "Mild shear is realistic; extreme changes shape unnaturally"),
]

print("=" * 65)
print("EXERCISE 25 — AUGMENTATION POLICY FOR PET BREED CLASSIFICATION")
print("=" * 65)
print(f"\n{'Augmentation':<30} {'Decision':<8} Reason")
print("-" * 95)
for aug, decision, reason in policy:
    print(f"  {aug:<28} [{decision:<5}] {reason}")

# ── Visualise policy as table ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
fig.suptitle("Exercise 25 — Augmentation Policy: Oxford-IIIT Pet Breed Classification",
             fontsize=12, fontweight="bold", y=0.98)

col_labels = ["Augmentation", "Decision", "Reason (label-preservation analysis)"]
table_data  = [[a, d, r] for a, d, r in policy]
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width([0, 1, 2])

# Colour rows by decision
colour_map = {"YES": "#c8e6c9", "NO": "#ffcdd2", "MAYBE": "#fff9c4"}
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#37474f")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 1:
        decision_val = policy[row - 1][1]
        cell.set_facecolor(colour_map.get(decision_val, "white"))
    cell.set_edgecolor("#cccccc")

patches = [mpatches.Patch(color="#c8e6c9", label="YES — safe augmentation"),
           mpatches.Patch(color="#ffcdd2", label="NO — destroys label"),
           mpatches.Patch(color="#fff9c4", label="MAYBE — use with care")]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=3, fontsize=9)

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_25_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")

# Recommended final policy
print("\n[Recommended production policy for pet breed classification]")
recommended = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.05, 0.2)),
])
print("  Compose([")
print("    RandomResizedCrop(224, scale=(0.7,1.0)),")
print("    RandomHorizontalFlip(p=0.5),")
print("    RandomRotation(degrees=15),")
print("    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),")
print("    GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),")
print("    ToTensor(),")
print("    RandomErasing(p=0.3, scale=(0.05,0.2))")
print("  ])")
print("Done ✓")
