"""
Exercise 09 — Convert RGB to HSV
==================================
Data:        Synthetic colourful RGB image
Program:     PIL RGB → OpenCV BGR → cv2.cvtColor(BGR, HSV)
Observation: HSV separates Hue (colour), Saturation (vividness), Value (brightness).
What Learned: HSV is more intuitive for colour-based operations than RGB.
Practical Use: Colour-based object segmentation (e.g., traffic cones, skin tone).
Limitation:  Hue wraps at red (0°/360°) — masks may need two ranges.
Next Question: How can we use HSV to detect objects of a specific colour?
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "exercise_09")
os.makedirs(OUT, exist_ok=True)

# Colourful synthetic image (hue wheel)
h, w = 128, 128
rgb = np.zeros((h, w, 3), dtype=np.uint8)
for col in range(w):
    hue_deg = col / w * 360
    # HSV → RGB manually for demonstration
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue_deg / 360, 1.0, 1.0)
    rgb[:, col, 0] = int(r * 255)
    rgb[:, col, 1] = int(g * 255)
    rgb[:, col, 2] = int(b * 255)

# Add a brightness gradient vertically
for row in range(h):
    factor = row / h
    rgb[row, :, :] = (rgb[row, :, :] * factor).astype(np.uint8)

print("=" * 55)
print("EXERCISE 09 — RGB TO HSV CONVERSION")
print("=" * 55)
print(f"\nRGB image shape: {rgb.shape}")

# Convert: PIL RGB → OpenCV BGR → HSV
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
print(f"\nHSV image shape: {hsv.shape}")
print(f"\n  H channel — range in OpenCV: 0–179 (maps to 0°–360°)")
print(f"              min: {H.min()}  max: {H.max()}  mean: {H.mean():.1f}")
print(f"  S channel — 0 (grey) to 255 (fully saturated)")
print(f"              min: {S.min()}  max: {S.max()}  mean: {S.mean():.1f}")
print(f"  V channel — 0 (black) to 255 (bright)")
print(f"              min: {V.min()}  max: {V.max()}  mean: {V.mean():.1f}")

sample_pixel = (64, 32)
r0, g0, b0 = rgb[sample_pixel[0], sample_pixel[1]]
h0, s0, v0 = hsv[sample_pixel[0], sample_pixel[1]]
print(f"\nSample pixel at {sample_pixel}:")
print(f"  RGB → R={r0}, G={g0}, B={b0}")
print(f"  HSV → H={h0} ({h0*2}°), S={s0}, V={v0}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Exercise 09 — RGB → HSV Conversion", fontsize=13, fontweight="bold")
axes[0].imshow(rgb);              axes[0].set_title("RGB original"); axes[0].axis("off")
axes[1].imshow(H, cmap="hsv");   axes[1].set_title("H (Hue)\n0–179 → 0°–360°"); axes[1].axis("off")
axes[2].imshow(S, cmap="gray");  axes[2].set_title("S (Saturation)\n0=grey, 255=vivid"); axes[2].axis("off")
axes[3].imshow(V, cmap="gray");  axes[3].set_title("V (Value/Brightness)\n0=black, 255=bright"); axes[3].axis("off")

plt.tight_layout()
out_fig = os.path.join(OUT, "exercise_09_output.png")
plt.savefig(out_fig, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {out_fig}")
print("Done ✓")
