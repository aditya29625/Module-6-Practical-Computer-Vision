# Module 6 — Practical Computer Vision

## Assignment Submission | Exercises 1–36 + Final Project Evaluation + Rapid Oral Evaluation

---

## Setup

```bash
pip install -r requirements.txt
```

## Run All Exercises

```bash
cd Module-6-Practical-Computer-Vision
python3 scripts/exercise_01.py   # Inspect Image as Numbers
# ... up to ...
python3 scripts/exercise_36.py   # CNN Architecture Thought Experiment
```

Or run all at once:

```bash
for i in $(seq -w 1 36); do python3 scripts/exercise_${i}.py; done
```

## Structure

```
Module-6-Practical-Computer-Vision/
│
├── README.md               ← this file
├── requirements.txt        ← pip dependencies
├── .gitignore
│
├── scripts/
│   ├── pet_helper.py       ← shared synthetic pet image generator
│   ├── exercise_01.py      ← Inspect Image as Numbers
│   ├── exercise_02.py      ← Vertical Edge Detector
│   ├── exercise_03.py      ← Horizontal Edge Detector
│   ├── exercise_04.py      ← Sobel X/Y + Gradient Magnitude
│   ├── exercise_05.py      ← Averaging Blur
│   ├── exercise_06.py      ← Gaussian Blur + Canny
│   ├── exercise_07.py      ← Image Sharpening
│   ├── exercise_08.py      ← Inspect RGB Channels
│   ├── exercise_09.py      ← RGB → HSV Conversion
│   ├── exercise_10.py      ← Green Object Detection (HSV Mask)
│   ├── exercise_11.py      ← Horizontal Flip
│   ├── exercise_12.py      ← Vertical Flip
│   ├── exercise_13.py      ← Rotation (~20°)
│   ├── exercise_14.py      ← Translation (RandomAffine)
│   ├── exercise_15.py      ← Scale (RandomAffine)
│   ├── exercise_16.py      ← Random Crop (RandomResizedCrop)
│   ├── exercise_17.py      ← Brightness Augmentation
│   ├── exercise_18.py      ← Contrast Augmentation
│   ├── exercise_19.py      ← Saturation Augmentation
│   ├── exercise_20.py      ← Hue Augmentation
│   ├── exercise_21.py      ← Blur as Augmentation
│   ├── exercise_22.py      ← Perspective Distortion
│   ├── exercise_23.py      ← Random Erasing / Occlusion
│   ├── exercise_24.py      ← Combined Augmentation Pipeline
│   ├── exercise_25.py      ← Augmentation Policy Design
│   ├── exercise_26.py      ← Learnable Convolution (nn.Conv2d)
│   ├── exercise_27.py      ← Pass Data Through Convolution
│   ├── exercise_28.py      ← Padding
│   ├── exercise_29.py      ← Stride
│   ├── exercise_30.py      ← ReLU Activation
│   ├── exercise_31.py      ← Max Pooling
│   ├── exercise_32.py      ← Build a Tiny CNN
│   ├── exercise_33.py      ← Classification Output (Logits → Softmax)
│   ├── exercise_34.py      ← Accuracy Can Mislead
│   ├── exercise_35.py      ← Confusion Matrix
│   └── exercise_36.py      ← CNN Architecture Thought Experiment
│
├── outputs/
│   ├── exercise_01/        ← output images for each exercise
│   ├── ...
│   └── exercise_36/
│
└── docs/
    └── exercises_QA.md     ← Full Q&A for all 36 exercises + Final Project + Oral Eval
```

## Notes on Dataset

This submission uses **synthetically generated images** for all exercises so the code runs
without internet access or dataset downloads. The augmentation logic (Exercises 11–25) is
identical to what you would use with Oxford-IIIT Pet images — just swap `get_pet_image()` with
`PIL.Image.open("path/to/pet.jpg")`.

## Dependencies

- `numpy` — array operations, kernel math
- `pillow` — image loading/saving
- `opencv-python` — classical CV operations
- `matplotlib` — visualisations
- `scikit-learn` — confusion matrix, classification report
- `torch` / `torchvision` — CNN building blocks, augmentation transforms
