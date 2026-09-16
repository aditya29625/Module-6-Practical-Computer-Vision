"""
Exercise 10 — Detect Green Objects with HSV Mask
==================================================
Data:        Synthetic RGB image with a green region
Program:     RGB → BGR → HSV → apply green mask [35,50,50]–[85,255,255]
Observation: Binary mask isolates green pixels; masked output shows only green region.
What Learned: HSV colour thresholding is a fast, deterministic way to detect coloured objects.
Practical Use: Traffic light detection, plant health analysis, sports tracking.
Limitation:  Fails under different lighting; overlapping hues cause false positives.
Next Question: Can we use contours on the mask to find the bounding box of the green object?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_10")
os.makedirs(OUT, exist_ok=True)

# Synthetic image: green circle on multi-colour background
h, w = 128, 128
rgb = np.zeros((h, w, 3), dtype=np.uint8)
# Red patch
rgb[0:64, 0:64] = [200, 50, 50]
# Blue patch
rgb[0:64, 64:] = [50, 80, 200]
# Yellow patch
rgb[64:, 64:] = [200, 200, 50]
# Purple patch
rgb[64:, 0:64] = [150, 50, 170]
# Green circle
cv2.circle(rgb, (64, 64), 28, (30, 200, 60), -1)

print("=" * 55)
print("EXERCISE 10 — DETECT GREEN OBJECTS WITH HSV MASK")
print("=" * 55)

# Convert RGB → BGR → HSV
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Green range in OpenCV HSV (H: 35–85, S: 50–255, V: 50–255)
lower = np.array([35, 50, 50])
upper = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

# Masked output (keep only green pixels)
masked = cv2.bitwise_and(rgb, rgb, mask=mask)

green_pixels = np.count_nonzero(mask)
print(f"\nTotal pixels   : {h*w}")
print(f"Green pixels   : {green_pixels}")
print(f"Green fraction : {green_pixels/(h*w)*100:.1f}%")
print(f"\nHSV lower bound: H={lower[0]}, S={lower[1]}, V={lower[2]}")
print(f"HSV upper bound: H={upper[0]}, S={upper[1]}, V={upper[2]}")
print(f"  → H range 35–85 corresponds to ~70°–170° (yellow-green to cyan-green)")

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rgb_annotated = rgb.copy()
if contours:
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(rgb_annotated, (x, y), (x+cw, y+ch), (255, 255, 0), 2)
    print(f"\nLargest green object bounding box: x={x}, y={y}, w={cw}, h={ch}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 10 — Green Object Detection via HSV Mask", fontsize=13, fontweight="bold")
axes[0].imshow(rgb);            axes[0].set_title("Original RGB"); axes[0].axis("off")
axes[1].imshow(hsv[:,:,0], cmap="hsv"); axes[1].set_title("Hue channel"); axes[1].axis("off")
axes[2].imshow(mask, cmap="gray");      axes[2].set_title(f"Green mask\n({green_pixels} pixels)"); axes[2].axis("off")
axes[3].imshow(rgb_annotated);          axes[3].set_title("Original + bounding box"); axes[3].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_10_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
