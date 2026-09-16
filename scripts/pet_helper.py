"""
Shared helper: generate a synthetic "pet-like" image for augmentation exercises.
Used by exercises 11–25.
"""
import numpy as np
from PIL import Image
import os

def get_pet_image(size=(224, 224)):
    """Generate a colourful synthetic image that resembles a pet photo."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background gradient (sky blue)
    for row in range(h):
        blue = int(180 + (row / h) * 50)
        img[row, :] = [max(0, 220 - row//3), max(0, 200 - row//4), blue]
    # Ground (green)
    img[h*2//3:, :] = [80, 160, 60]
    # Body (orange/brown ellipse) — "cat body"
    cx, cy = w//2, h*2//3 - 10
    for y in range(h):
        for x in range(w):
            if ((x-cx)**2)/2500 + ((y-cy)**2)/1200 < 1:
                img[y, x] = [210, 140, 80]
    # Head (circle)
    hcx, hcy = w//2, h//2 - 10
    for y in range(h):
        for x in range(w):
            if (x-hcx)**2 + (y-hcy)**2 < 900:
                img[y, x] = [215, 150, 90]
    # Eyes
    for ey, ex in [(hcy-5, hcx-12), (hcy-5, hcx+12)]:
        for y in range(h):
            for x in range(w):
                if (x-ex)**2 + (y-ey)**2 < 25:
                    img[y, x] = [30, 20, 15]
    # Nose
    for y in range(h):
        for x in range(w):
            if (x-hcx)**2 + (y-hcy-8)**2 < 12:
                img[y, x] = [200, 80, 80]
    return Image.fromarray(img, mode="RGB")

if __name__ == "__main__":
    img = get_pet_image()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "synthetic_pet.png")
    img.save(out)
    print(f"Saved synthetic pet image → {out}")
