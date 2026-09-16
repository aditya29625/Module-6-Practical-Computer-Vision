"""
Exercise 07 — Sharpen an Image
================================
Data:        Synthetic blurry image
Program:     Apply sharpening kernel via cv2.filter2D
Observation: Fine details and edges appear more pronounced.
What Learned: Sharpening = original + scaled Laplacian (high-frequency enhancement).
Practical Use: Medical imaging, print pre-processing, photography.
Limitation:  Amplifies noise along with edges; over-sharpening creates halos.
Next Question: How can we sharpen selectively — only in smooth regions?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_07")
os.makedirs(OUT, exist_ok=True)

# Build a moderately blurry base image
img = np.zeros((128, 128), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (108, 108), 200, -1)
cv2.circle(img, (64, 64), 30, 100, -1)
for i in range(0, 128, 16):
    img[i:i+8, :] = np.maximum(img[i:i+8, :], 50)

# Pre-blur to simulate a soft/blurry input
blurry = cv2.GaussianBlur(img, (5, 5), 0)

# Sharpening kernel
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)

print("=" * 55)
print("EXERCISE 07 — SHARPEN AN IMAGE")
print("=" * 55)
print(f"\nSharpening kernel:\n{sharpen_kernel}")
print("\nInterpretation: centre weight=5 amplifies current pixel,")
print("neighbours=-1 subtract blur → emphasises high-frequency detail")

sharpened = cv2.filter2D(blurry, -1, sharpen_kernel)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Edge energy comparison
def edge_energy(im):
    gx = cv2.Sobel(im, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(im, cv2.CV_64F, 0, 1)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))

print(f"\nEdge energy (blurry)    : {edge_energy(blurry):.2f}")
print(f"Edge energy (sharpened) : {edge_energy(sharpened):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Exercise 07 — Image Sharpening", fontsize=13, fontweight="bold")
axes[0].imshow(blurry,    cmap="gray"); axes[0].set_title("Blurry input");  axes[0].axis("off")
axes[1].imshow(sharpened, cmap="gray"); axes[1].set_title("Sharpened output"); axes[1].axis("off")
diff = np.abs(sharpened.astype(int) - blurry.astype(int)).astype(np.uint8)
axes[2].imshow(diff, cmap="hot"); axes[2].set_title("Difference\n(added detail)"); axes[2].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_07_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
