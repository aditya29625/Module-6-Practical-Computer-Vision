# Module 6: Practical Computer Vision — Exercises Q&A, Final Project & Oral Evaluation

---

# PART 1: COMPLETE EXERCISES 1–36

---

## Exercise 01 — Inspect Image as Numbers

* **Data**: Deterministically generated synthetic test images (Grayscale 128×128 horizontal gradient and RGB 128×128 color gradient).
* **Program**: Loaded image via Pillow (`Image.open().convert()`), converted to NumPy array (`np.array()`), inspected `.shape`, `.dtype`, min/max values, slice of pixel values, and normalized to range $[0.0, 1.0]$.
* **Observation**:
  * Grayscale shape: `(128, 128)` — 2D matrix of 8-bit integers ranging $0 \dots 255$.
  * RGB shape: `(128, 128, 3)` — 3D tensor with height, width, and 3 color channels.
  * Normalizing via `/ 255.0` shifts range from $[0, 255]$ to $[0.0, 1.0]$ with float64/float32 precision.
* **What We Learned**: An image is purely a mathematical grid/tensor of numbers. Computer vision algorithms do not "see" pictures; they perform arithmetic over numerical matrices.
* **Practical Use**: Standard entry point for any visual ingestion pipeline (preprocessing, normalization, dataset sanity checks).
* **Limitation**: Direct pixel inspection alone does not reveal spatial semantics or object boundaries.
* **Next Question**: How can we detect structural transitions (edges) by computing differences between adjacent numbers?

### Teacher's Evaluation Questions & Answers
1. *Why are grayscale images represented as 2D arrays while RGB images are 3D arrays?*
   **Answer**: Grayscale images have only one intensity channel per coordinate $(x, y)$, giving shape $(H, W)$. RGB images have three intensity values (Red, Green, Blue) per pixel coordinate $(x, y)$, yielding shape $(H, W, 3)$.
2. *Why do we divide pixel values by 255.0 before feeding them into deep neural networks?*
   **Answer**: Dividing by 255 scales integers from $[0, 255]$ into floating-point numbers in $[0.0, 1.0]$. This prevents vanishing or exploding gradients, ensures numerical stability, and matches initial weight scales during backpropagation.

---

## Exercise 02 — Build a Vertical Edge Detector

* **Data**: Synthetic $128 \times 128$ image containing vertical rectangular intensity steps.
* **Program**: Applied vertical edge kernel using 2D cross-correlation / convolution (`cv2.filter2D`):
  $$\begin{bmatrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{bmatrix}$$
* **Observation**: Left-to-right transitions produce high absolute responses; uniform regions produce 0.
* **What We Learned**: A convolution kernel computes a localized difference operator. Columns with $[-1, 0, +1]$ compute horizontal gradient $\frac{\partial I}{\partial x}$, which highlights vertical boundaries.
* **Practical Use**: Detecting building pillars, lane boundaries, bar codes, or vertical document borders.
* **Limitation**: Sensitive to high-frequency sensor noise; completely ignores purely horizontal edges.
* **Next Question**: How do we modify the kernel matrix to detect horizontal boundaries instead?

### Teacher's Evaluation Questions & Answers
1. *Why does the sum of the weights in an edge detection kernel usually equal 0?*
   **Answer**: When the kernel passes over a flat, homogeneous region (where all pixel values are equal), the sum of products $(-1 \cdot p + 0 \cdot p + 1 \cdot p)$ equals 0. A zero sum ensures that uniform regions produce zero response and only contrast differences trigger activation.
2. *What does a negative value in the filtered response signify?*
   **Answer**: A negative response indicates a transition from light to dark along the direction of the derivative (e.g., bright pixel on left, dark pixel on right). Taking the absolute value (`np.abs`) allows detecting edges regardless of transition polarity.

---

## Exercise 03 — Design a Horizontal Edge Detector

* **Data**: Synthetic $128 \times 128$ image containing horizontal bar structures.
* **Program**: Implemented horizontal kernel using `cv2.filter2D`:
  $$\begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{bmatrix}$$
* **Observation**: Strong activations at top and bottom horizontal boundaries; zero activation along vertical lines.
* **What We Learned**: Rotating the kernel 90 degrees switches sensitivity to orthogonal spatial gradients ($\frac{\partial I}{\partial y}$).
* **Practical Use**: Detecting horizon lines, road markings, table borders, and shelving rows.
* **Limitation**: Misses vertical edges entirely and produces weakened responses on 45° diagonals.
* **Next Question**: How can we combine horizontal and vertical detectors to measure overall edge strength in any arbitrary direction?

### Teacher's Evaluation Questions & Answers
1. *How does the horizontal kernel differ mathematically from the vertical kernel?*
   **Answer**: The horizontal kernel is the transpose ($K_h = K_v^T$) of the vertical kernel. It computes differences across rows ($\Delta y$) rather than across columns ($\Delta x$).
2. *Can a horizontal edge detector detect diagonal edges?*
   **Answer**: Partially. Diagonal edges have non-zero gradient projections along both the $x$ and $y$ axes, so a horizontal kernel will produce a partial (attenuated) response proportional to $\sin(\theta)$.

---

## Exercise 04 — Sobel X and Sobel Y + Gradient Magnitude

* **Data**: Synthetic test image with circular, rectangular, and diagonal geometry.
* **Program**: Computed Sobel gradients $G_x = \text{SobelX}(I)$ and $G_y = \text{SobelY}(I)$, then combined them into gradient magnitude:
  $$\text{magnitude} = \sqrt{G_x^2 + G_y^2}$$
* **Observation**: Sobel X captured vertical edges, Sobel Y captured horizontal edges, and the gradient magnitude highlighted all boundaries omnidirectionally.
* **What We Learned**: Sobel filters incorporate smoothing weights $[1, 2, 1]$ orthogonal to the derivative direction, reducing noise sensitivity while computing derivatives.
* **Practical Use**: Feature extraction for classical computer vision (HOG descriptors, boundary detection).
* **Limitation**: Edges are thick and multi-pixel wide; thresholds must be selected manually.
* **Next Question**: How can we suppress noise before edge calculation to avoid spurious detections?

### Teacher's Evaluation Questions & Answers
1. *Why does the Sobel operator use weights $[1, 2, 1]$ in its smoothing axis instead of $[1, 1, 1]$?*
   **Answer**: The weights $[1, 2, 1]$ approximate a 1D Gaussian smoothing filter (derived from Pascal's triangle / binomial coefficients), giving higher importance to the central pixel and providing superior isotropic smoothing compared to a simple box filter.
2. *What additional information can be derived from $G_x$ and $G_y$ besides magnitude?*
   **Answer**: Edge orientation/direction can be computed as $\theta = \arctan2(G_y, G_x)$, which gives the exact angle perpendicular to the edge boundary.

---

## Exercise 05 — Blur with an Averaging Kernel

* **Data**: Synthetic image contaminated with Gaussian noise ($\sigma = 30$).
* **Program**: Convolved noisy image with a normalized $5 \times 5$ averaging box kernel:
  $$K_{\text{avg}} = \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}$$
* **Observation**: High-frequency noise variance was noticeably reduced; sharp boundaries became smoothed and blurred.
* **What We Learned**: An averaging filter is a low-pass spatial filter. The sum of weights must equal 1.0 to preserve overall image energy/brightness.
* **Practical Use**: Denoising background artifacts prior to global thresholding.
* **Limitation**: Box blur creates unnatural rectangular artifacts and destroys sharp edge definition.
* **Next Question**: How does Gaussian blur provide smoother, radially symmetric low-pass filtering?

### Teacher's Evaluation Questions & Answers
1. *Why must the kernel values sum to 1 in a blur filter?*
   **Answer**: If the kernel values sum to $>1$, the resulting image will become progressively brighter (energy amplification); if $<1$, it will darken. A sum of 1.0 preserves the mean luminance level.
2. *What happens to sharp edges when an averaging kernel size increases from $3\times3$ to $11\times11$?*
   **Answer**: Larger kernels incorporate pixels from further away, causing significant boundary smearing, loss of fine texture, and loss of small structural features.

---

## Exercise 06 — Gaussian Blur + Canny Edge Detection

* **Data**: Synthetic geometric scene with additive random noise.
* **Program**: Implemented complete multi-stage Canny pipeline:
  `cv2.GaussianBlur(arr, (5, 5), 0)` $\rightarrow$ `cv2.Canny(blurred, 100, 200)`.
* **Observation**: Produced thin, single-pixel wide, clean, continuous edge lines with zero noise artifacts.
* **What We Learned**: Canny combines 4 critical steps: (1) Gaussian smoothing, (2) Sobel gradient computation, (3) Non-Maximum Suppression (NMS) for thinning, and (4) Hysteresis thresholding with dual thresholds.
* **Practical Use**: Real-time contour analysis, industrial part inspection, lane line segmentation.
* **Limitation**: Hysteresis thresholds ($T_{\text{low}}, T_{\text{high}}$) require empirical tuning under changing illumination.
* **Next Question**: Can we do the reverse of blurring — enhance and sharpen fine details?

### Teacher's Evaluation Questions & Answers
1. *Explain the role of the two thresholds (e.g., 100 and 200) in Canny edge detection.*
   **Answer**: Pixels with gradient magnitude $>200$ are classified as strong edges (accepted immediately). Pixels $<100$ are rejected. Pixels between $100$ and $200$ (weak edges) are accepted only if they are spatially connected to strong edge pixels, preventing edge fragmentation while rejecting isolated noise.
2. *What is Non-Maximum Suppression (NMS) in Canny?*
   **Answer**: NMS checks whether the gradient magnitude at a pixel is the local maximum along the direction of the gradient vector. If not, it suppresses the pixel to 0, ensuring edges are exactly 1 pixel thick.

---

## Exercise 07 — Sharpen an Image

* **Data**: Blurred synthetic image.
* **Program**: Convolved with a 2D sharpening Laplacian-based kernel:
  $$K_{\text{sharp}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
* **Observation**: Blurred contours regained crispness, contrast at boundaries increased, and edge energy metric rose.
* **What We Learned**: Sharpening adds high-frequency components back to the image ($I_{\text{sharp}} = I + \alpha \cdot \nabla^2 I$). The center weight $5 = 1 + 4$ preserves the identity while subtracting the average of surrounding pixels.
* **Practical Use**: Enhancing medical X-rays, satellite imagery, and forensic image analysis.
* **Limitation**: High-frequency noise or JPEG compression artifacts are amplified along with edges.
* **Next Question**: How do color channels interact, and how do we examine individual color planes?

### Teacher's Evaluation Questions & Answers
1. *Why does the center value of the sharpening kernel equal 5 while the four neighbors equal -1?*
   **Answer**: The center pixel represents the original signal ($1.0$) plus a negative discrete Laplacian ($4 \cdot p - p_{\text{top}} - p_{\text{bottom}} - p_{\text{left}} - p_{\text{right}}$). Combining them yields $1 + 4 = 5$ at the center and $-1$ on each 4-connected neighbor. The sum of all elements is $5 - 4 = 1$, preserving average brightness.
2. *What is the visual consequence of over-sharpening an image?*
   **Answer**: Excessive sharpening causes high-contrast "halo" artifacts around object boundaries, clipped highlights/shadows, and amplified grain/sensor noise.

---

## Exercise 08 — Inspect RGB Channels

* **Data**: Synthetic RGB test image with distinct primary and composite color swatches.
* **Program**: Extracted individual channels (`R = img[:,:,0]`, `G = img[:,:,1]`, `B = img[:,:,2]`), analyzed shapes, value ranges, and displayed individual channel intensities.
* **Observation**: Each color channel is a 2D grayscale array $[0, 255]$. Pure red regions show high intensity in Channel 0 and near-zero in Channels 1 and 2.
* **What We Learned**: An RGB image is a linear superposition of three spectral bands. Color perception in computers relies on additive trichromatic synthesis.
* **Practical Use**: White balance calibration, checking sensor saturation per color band, basic channel thresholding.
* **Limitation**: In RGB space, chromaticity (color) and luminance (brightness) are heavily entangled across all 3 channels.
* **Next Question**: How can we decouple chromaticity from illumination using the HSV color space?

### Teacher's Evaluation Questions & Answers
1. *If a pixel has RGB values $(255, 255, 0)$, what color is perceived and which channels are active?*
   **Answer**: The color is yellow (additive blend of Red and Green). Channel 0 (Red) and Channel 1 (Green) are at maximum intensity ($255$), while Channel 2 (Blue) is $0$.
2. *Why is RGB generally unsuitable for color tracking under changing lighting conditions?*
   **Answer**: A change in illumination scales all three $(R, G, B)$ values simultaneously, shifting the Euclidean distance in RGB space even if the object's intrinsic surface color hasn't changed.

---

## Exercise 09 — Convert RGB to HSV

* **Data**: Synthetic multi-hue, variable-brightness color spectrum image.
* **Program**: Converted RGB $\rightarrow$ BGR $\rightarrow$ HSV (`cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)`), extracted Hue ($H$), Saturation ($S$), and Value ($V$) channels.
* **Observation**:
  * $H \in [0, 179]$ (OpenCV scales $360^\circ$ by $\frac{1}{2}$ to fit in `uint8`).
  * $S \in [0, 255]$ represents color purity/vividness.
  * $V \in [0, 255]$ represents perceived luminance/brightness.
* **What We Learned**: HSV decouples chromatic identity ($H$) from color dilution ($S$) and illumination level ($V$).
* **Practical Use**: Robust color segmentation unaffected by shadows or ambient lighting variations.
* **Limitation**: Hue is circular ($0^\circ \equiv 360^\circ$); red wraps around the boundary ($[0, 10]$ and $[170, 179]$).
* **Next Question**: How can we build an explicit binary mask to isolate specific colored objects using HSV ranges?

### Teacher's Evaluation Questions & Answers
1. *Why is Hue mapped to range $0 \dots 179$ in OpenCV instead of $0 \dots 359$?*
   **Answer**: Standard 8-bit unsigned integers (`uint8`) can only store values up to 255. To fit the $360^\circ$ angular color wheel into a single `uint8` byte, OpenCV divides the degrees by 2 ($360 / 2 = 180 \rightarrow [0, 179]$).
2. *What happens to the Hue value when a pixel has Saturation $= 0$?*
   **Answer**: When Saturation is 0, the pixel is an achromatic shade of gray (or black/white). In this state, Hue is mathematically undefined / arbitrary (often defaulting to 0).

---

## Exercise 10 — Detect Green Objects with a Mask

* **Data**: Synthetic multi-colored scene containing a distinct green circular target.
* **Program**: Converted to HSV, applied threshold range `lower = [35, 50, 50]`, `upper = [85, 255, 255]` with `cv2.inRange()`, performed bitwise AND (`cv2.bitwise_and`), and extracted object bounding box via `cv2.findContours`.
* **Observation**: Binary mask isolated green pixels with 100% precision; bounding box accurately enclosed the target.
* **What We Learned**: HSV thresholding produces an effective deterministic segmenter for color-coded industrial markers or items.
* **Practical Use**: Green screen keying, agricultural crop/weed detection, robotics ball tracking.
* **Limitation**: Fails if background objects share identical hue or if extreme lighting drops saturation below threshold.
* **Next Question**: How do we augment training data to make deep learning models invariant to changes in orientation, scale, and lighting?

### Teacher's Evaluation Questions & Answers
1. *What is the role of `cv2.inRange()` in color segmentation?*
   **Answer**: `cv2.inRange()` performs an element-wise threshold check: for each pixel, if $L_i \le P_i \le U_i$ across all channels ($H, S, V$), the output pixel is set to 255 (white); otherwise 0 (black), creating a binary mask.
2. *Why is bitwise AND used with the mask and the original image?*
   **Answer**: Bitwise AND preserves original RGB pixel values wherever mask pixels are $255$ ($1 \text{ AND } X = X$) and sets all other background regions to black ($0 \text{ AND } X = 0$).

---

## Exercise 11 — Horizontal Flip

* **Data**: Synthetic pet image ($224 \times 224 \times 3$).
* **Program**: Applied `torchvision.transforms.functional.hflip()`.
* **Observation**: The pet image is mirrored left-to-right; pixel column $x$ maps to $W - 1 - x$.
* **What We Learned**: Horizontal flipping introduces lateral pose variation while strictly preserving label semantics for natural objects.
* **Practical Use**: Standard baseline augmentation for ImageNet, CIFAR-10, Oxford Pets classification.
* **Limitation**: Invalid for character recognition (OCR, e.g., 'b' vs 'd'), traffic sign direction, or asymmetric medical scans.
* **Next Question**: Is vertical flipping equally safe for natural pet images?

### Teacher's Evaluation Questions & Answers
1. *Why is horizontal flip considered a "label-preserving" transformation for pet classification?*
   **Answer**: A dog or cat facing left has the exact same biological identity and breed class as one facing right. The semantic label remains invariant under reflection across the vertical axis.
2. *Under what circumstances does horizontal flipping destroy ground-truth labels?*
   **Answer**: In OCR/text reading (flipping 'E' or 'b'), road sign interpretation (left turn vs right turn), steering angle prediction in autonomous driving, or asymmetric organ identification in radiology.

---

## Exercise 12 — Vertical Flip

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.functional.vflip()`.
* **Observation**: Image is inverted upside-down; ground appears at top and sky at bottom.
* **What We Learned**: Vertical flipping alters gravity-oriented visual priors.
* **Practical Use**: Aerial/satellite imagery, overhead drone footage, microscopy cell imaging (where there is no canonical "up").
* **Limitation**: Highly unnatural for terrestrial photography; introduces out-of-distribution artifacts into pet models.
* **Next Question**: How does small continuous rotation differ from discrete 180° flipping?

### Teacher's Evaluation Questions & Answers
1. *Why should vertical flip generally be avoided for ground-level camera datasets?*
   **Answer**: Terrestrial objects conform to gravity: sky is above, ground is below, legs point downward. Training with vertically inverted pets forces the model to waste capacity learning impossible real-world orientations.
2. *Name two domains where vertical flip is a standard, valid augmentation.*
   **Answer**: (1) Satellite / remote sensing imagery, and (2) Histopathology / cell microscopy slide imaging.

---

## Exercise 13 — Rotation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.functional.rotate(img, angle=20)` with $\pm 20^\circ$ rotation angles.
* **Observation**: Image content rotated around center; corner voids filled with zero-padding (black pixels).
* **What We Learned**: Rotation teaches convolutional networks rotational tolerance without requiring full rotation-equivariant architectures.
* **Practical Use**: Mobile photo capture, drone camera monitoring, handheld scanning.
* **Limitation**: Aggressive rotation angles introduce large corner-fill artifacts or crop valid outer features.
* **Next Question**: How can we simulate shifts in camera framing and off-center subjects?

### Teacher's Evaluation Questions & Answers
1. *What happens to corner pixels during an image rotation if no cropping is applied?*
   **Answer**: Empty triangular regions are created at the four image corners where image content no longer exists. These regions are typically filled with constant black pixels ($0$) or reflective border padding.
2. *Why is a $\pm 15^\circ$ to $\pm 20^\circ$ rotation angle range typically chosen for pet datasets rather than $\pm 90^\circ$?*
   **Answer**: Real pets tilt their heads or lean slightly within $\pm 20^\circ$. A $90^\circ$ tilt is rare, and extreme rotations distort natural vertical posture priors.

---

## Exercise 14 — Translation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))`.
* **Observation**: Image shifted randomly along $X$ and $Y$ dimensions by up to 20% of spatial dimensions.
* **What We Learned**: Translation forces the model to locate discriminative features regardless of their coordinate location within the frame.
* **Practical Use**: Object tracking, casual photography where subjects are not perfectly centered.
* **Limitation**: Excessive translation pushes critical features completely out of the visible receptive window.
* **Next Question**: How do we simulate distance variations where the subject appears larger or smaller?

### Teacher's Evaluation Questions & Answers
1. *Why are standard CNNs not automatically fully translation-invariant?*
   **Answer**: While convolution is translation-equivariant (shifting input shifts feature map), pooling and boundary padding introduce spatial phase sensitivity, and fully connected classification heads require feature spatial invariance learned from data.
2. *What parameter controls the maximum horizontal and vertical shift in PyTorch's `RandomAffine`?*
   **Answer**: The `translate=(a, b)` tuple, where $a$ is the maximum horizontal fraction and $b$ is the maximum vertical fraction of the image dimensions.

---

## Exercise 15 — Scale (Zoom)

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.RandomAffine(degrees=0, scale=(0.7, 1.3))`.
* **Observation**: Content zoomed in (magnifying texture) or zoomed out (revealing padding border) while maintaining fixed canvas dimensions ($224 \times 224$).
* **What We Learned**: Scale jittering trains multi-scale feature representations, preparing filters for objects at variable distances.
* **Practical Use**: Security surveillance cameras, long-range wildlife photography.
* **Limitation**: Downscaling reduces spatial resolution; excessive upscaling causes bilinear interpolation blur.
* **Next Question**: How does RandomResizedCrop combine scaling and aspect ratio jittering into a single transform?

### Teacher's Evaluation Questions & Answers
1. *What is the difference between scale augmentation and resizing an image?*
   **Answer**: Standard resizing scales the entire image to fixed dimensions deterministically. Scale augmentation randomly crops or zooms the object at varying magnification factors during each training epoch.
2. *How does scale variation help deep CNN feature extractors?*
   **Answer**: It forces early and intermediate layers to recognize patterns (e.g., whiskers, eyes) across different receptive field sizes and spatial frequencies.

---

## Exercise 16 — Random Crop (RandomResizedCrop)

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33))`.
* **Observation**: Extracted random rectangular sub-regions (50%–100% area) and resized them back to $224 \times 224$.
* **What We Learned**: RandomResizedCrop is the cornerstone augmentation of modern vision training (ImageNet recipe); it forces the model to recognize objects from partial/local cues.
* **Practical Use**: Standard baseline for ResNet, ConvNeXt, Vision Transformers.
* **Limitation**: May accidentally crop out the primary object if `scale` minimum is set too low ($<0.1$).
* **Next Question**: How can we simulate environmental lighting changes like overcast skies or direct sunlight?

### Teacher's Evaluation Questions & Answers
1. *Why is `RandomResizedCrop` preferred over simple `CenterCrop` during model training?*
   **Answer**: `CenterCrop` is deterministic and always presents the exact same framing, whereas `RandomResizedCrop` generates infinite spatial variations, preventing overfitting and forcing reliance on sub-features.
2. *What risk exists if the lower bound of `scale` in `RandomResizedCrop` is set to 0.05?*
   **Answer**: The crop may capture only an uninformative background texture (e.g., a patch of grass or wall) with zero pet pixels, but the training loss will still penalize it with the original label, injecting noisy gradients.

---

## Exercise 17 — Brightness Augmentation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.ColorJitter(brightness=0.5)`.
* **Observation**: Pixel intensities scaled by random factor $B \in [1 - 0.5, 1 + 0.5] = [0.5, 1.5]$.
* **What We Learned**: Brightness jitter adds uniform multiplicative/additive shifts to the luminance channel without modifying spatial structure.
* **Practical Use**: Ensuring outdoor camera algorithms work seamlessly from dawn to bright midday.
* **Limitation**: High values cause white pixel saturation ($255$ clipping); low values collapse into near-black shadows.
* **Next Question**: How does contrast modification differ from uniform brightness scaling?

### Teacher's Evaluation Questions & Answers
1. *How does `ColorJitter(brightness=0.5)` mathematically modify pixel values?*
   **Answer**: It multiplies pixel intensity values by a factor chosen uniformly from $[1 - 0.5, 1 + 0.5] = [0.5, 1.5]$, clipping the results to $[0, 255]$.
2. *Does brightness augmentation affect edge detection filters like Sobel?*
   **Answer**: Yes; because Sobel computes linear differences, scaling intensity by $k$ scales gradient magnitude by $k$. Normalization or contrast invariance helps mitigate this.

---

## Exercise 18 — Contrast Augmentation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.ColorJitter(contrast=0.5)`.
* **Observation**: Interpolated/extrapolated pixel intensities relative to the grayscale mean; low contrast looks foggy/gray, high contrast shows pronounced dynamic range.
* **What We Learned**: Contrast transforms adjust the spread of the pixel histogram without shifting the overall color tone.
* **Practical Use**: Autonomous driving in foggy, hazy, or rainy weather.
* **Limitation**: Extreme low contrast removes all edge gradients; extreme high contrast blows out dynamic range.
* **Next Question**: How do we adjust color vividness independently of contrast?

### Teacher's Evaluation Questions & Answers
1. *What is the mathematical formulation of contrast adjustment?*
   **Answer**: $I_{\text{contrast}} = \text{mean}(I_{\text{gray}}) + c \cdot (I - \text{mean}(I_{\text{gray}}))$, where $c$ is the contrast factor.
2. *Why is contrast augmentation useful for medical imagery?*
   **Answer**: Different scanners and patient tissue densities produce varying dynamic ranges; contrast jitter prevents models from relying on absolute intensity thresholds.

---

## Exercise 19 — Saturation Augmentation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.ColorJitter(saturation=0.5)`.
* **Observation**: Saturation factor $S \in [0.5, 1.5]$ scaled the chromatic purity; factor 0 produced pure grayscale.
* **What We Learned**: Modulating saturation tests whether a model relies excessively on vivid color versus structural morphology.
* **Practical Use**: Processing archival photos, cheap webcams, or varying outdoor ambient lighting.
* **Limitation**: Complete desaturation ($S=0$) destroys color-specific diagnostic features.
* **Next Question**: How does rotating the color wheel (Hue jitter) impact classification?

### Teacher's Evaluation Questions & Answers
1. *What happens to an RGB image when its saturation is reduced to 0?*
   **Answer**: The image becomes completely grayscale (monochromatic), where $R = G = B$ for every pixel.
2. *Is saturation augmentation safe for pet breed classification?*
   **Answer**: Yes, because most cat and dog breeds are identifiable by bone structure, fur pattern, facial geometry, and ear morphology rather than high-precision saturation levels.

---

## Exercise 20 — Hue Augmentation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.ColorJitter(hue=0.1)`.
* **Observation**: Hue shifted by up to $\pm 36^\circ$ ($\pm 0.1 \times 360^\circ$), altering color cast (orange fur shifted toward reddish or yellowish).
* **What We Learned**: Hue jitter shifts the angular coordinate on the color wheel. Small shifts simulate color temperature differences (warm vs cool lighting).
* **Practical Use**: White balance shift invariance for general object classification.
* **Limitation**: **DANGEROUS** if color defines the target class label (e.g., golden retriever vs black lab, traffic lights).
* **Next Question**: How can we make a model robust to camera defocus or motion blur?

### Teacher's Evaluation Questions & Answers
1. *Why is Hue jitter parameter restricted to $[-0.5, 0.5]$ in torchvision?*
   **Answer**: Because Hue is an angle on a $360^\circ$ circle, a shift of $0.5$ corresponds to $180^\circ$, which is the maximum possible opposite color inversion (e.g., blue $\rightarrow$ yellow).
2. *Give an example of a computer vision task where Hue augmentation is strictly prohibited.*
   **Answer**: Traffic light recognition (turns red into green), skin lesion malignancy classification (melanoma color is diagnostic), or chemical flame test identification.

---

## Exercise 21 — Blur as Augmentation

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))`.
* **Observation**: High-frequency fur texture was attenuated; overall shapes remained recognizable.
* **What We Learned**: Blur augmentation prevents deep neural networks from becoming overly dependent on high-frequency pixel textures (combating texture bias).
* **Practical Use**: Processing low-resolution video feeds, moving cameras, sports analytics.
* **Limitation**: Too much blur destroys key diagnostic features like eye shape or facial contours.
* **Next Question**: How do we simulate non-frontal camera angles and perspective tilt?

### Teacher's Evaluation Questions & Answers
1. *What is "texture bias" in CNNs and how does Gaussian blur augmentation help mitigate it?*
   **Answer**: CNNs often rely heavily on fine surface texture (e.g., fur patterns) rather than global shape. Blur augmentation suppresses fine textures, forcing the network to learn global structural and shape representations.
2. *Why must the kernel size in `GaussianBlur` be an odd positive integer?*
   **Answer**: An odd kernel size (e.g., $3, 5, 7$) ensures a well-defined unique center pixel for symmetric spatial convolution.

---

## Exercise 22 — Perspective Distortion

* **Data**: Synthetic pet image.
* **Program**: Applied `torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=1.0)`.
* **Observation**: 4-point homography warped the square canvas into a trapezoid, simulating non-perpendicular viewpoints.
* **What We Learned**: Perspective transformation models non-affine projective homographies ($3 \times 3$ matrix) typical of wide-angle or angled lenses.
* **Practical Use**: License plate recognition (ALPR), document scanning apps, drone monitoring.
* **Limitation**: Extreme distortion creates severe geometric shearing that ruins shape geometry.
* **Next Question**: How can we force a network to recognize objects when parts are physically covered or occluded?

### Teacher's Evaluation Questions & Answers
1. *What is the mathematical difference between an affine transform and a perspective (projective) transform?*
   **Answer**: An affine transform maintains parallelism of lines (6 degrees of freedom, $2\times3$ matrix), whereas a perspective transform (homography, 8 degrees of freedom, $3\times3$ matrix) allows parallel lines to converge at vanishing points.
2. *When is perspective distortion especially critical in real-world deployment?*
   **Answer**: In fixed CCTV security cameras and mobile document scanning, where objects are viewed from oblique, non-orthogonal angles.

---

## Exercise 23 — Random Erasing / Occlusion

* **Data**: Synthetic pet image converted to PyTorch tensor.
* **Program**: Applied `torchvision.transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), value=0)`.
* **Observation**: A random rectangular patch (10%–30% of image area) was replaced with black zero-pixels.
* **What We Learned**: Random erasing acts as spatial dropout, preventing co-adaptation of features and forcing the network to attend to multiple redundant visual cues.
* **Practical Use**: Person re-identification, face recognition with masks/glasses, cluttered environments.
* **Limitation**: May erase the entire discriminating feature if the target object is very small.
* **Next Question**: How do we orchestrate all these individual transformations into a single unified training pipeline?

### Teacher's Evaluation Questions & Answers
1. *How does `RandomErasing` relate to the Dropout regularization technique?*
   **Answer**: While Dropout randomly zeroes out intermediate activations in feature space, RandomErasing zeroes out contiguous spatial regions in input pixel space (spatial dropout), preventing reliance on single localized features.
2. *Why does `RandomErasing` typically run after `ToTensor()` rather than on PIL images?*
   **Answer**: `RandomErasing` operates directly on multi-dimensional tensors ($C, H, W$) to efficiently perform in-place memory replacement with tensor constants or noise distributions.

---

## Exercise 24 — Combine Augmentations

* **Data**: Synthetic pet image.
* **Program**: Built sequential `torchvision.transforms.Compose` pipeline:
  `RandomResizedCrop` $\rightarrow$ `RandomHorizontalFlip` $\rightarrow$ `RandomRotation` $\rightarrow$ `ColorJitter` $\rightarrow$ `RandomPerspective`.
* **Observation**: Executing the pipeline 8 independent times produced 8 completely distinct, visually realistic training examples.
* **What We Learned**: Stochastic composition of geometric and photometric augmentations expands a small training set into a virtually infinite variety of training samples.
* **Practical Use**: Production training pipelines for PyTorch vision models.
* **Limitation**: Pipeline execution overhead can become a CPU bottleneck if transforms are not optimized or parallelized.
* **Next Question**: How do we systematically decide which transforms to include or exclude for a specific dataset?

### Teacher's Evaluation Questions & Answers
1. *Why should stochastic data augmentation only be applied to the training set and not the validation/test sets?*
   **Answer**: Training requires varied samples to generalize and regularize weights, while validation/testing must evaluate the model deterministically on real, unaltered data to provide an unbiased benchmark.
2. *What is test-time augmentation (TTA)?*
   **Answer**: TTA generates multiple augmented versions of a single test image at inference time, passes each through the model, and averages the resulting probability predictions to improve accuracy and robustness.

---

## Exercise 25 — Design an Augmentation Policy

* **Data**: Oxford-IIIT Pet Dataset (37 classes: cats and dogs).
* **Program**: Developed a domain-specific augmentation policy matrix categorizing all augmentations as **YES / NO / MAYBE** with rigorous label-preservation analysis.
* **Observation**: Geometric shifts, horizontal flips, and moderate color jitter are safe; vertical flips and aggressive hue rotations are detrimental.
* **What We Learned**: An augmentation policy must be tailored to task semantics. A transform is valid if and only if a human expert assigns the exact same label to the transformed image.
* **Practical Use**: Establishing data contracts and training configs for computer vision projects.
* **Limitation**: Manual policy design requires domain expertise (Automated alternatives include AutoAugment / RandAugment).
* **Next Question**: How do learnable convolutional layers extract features from these images?

### Summary Policy Table for Pet Classification
| Augmentation | Policy | Technical Justification |
| :--- | :--- | :--- |
| **Horizontal Flip** | **YES** | Bilateral animal symmetry; dogs/cats look identical left vs right. |
| **Vertical Flip** | **NO** | Inverts gravity; animals do not stand on ceilings in real test scenarios. |
| **Rotation ($\pm 15^\circ$)** | **YES** | Accounts for natural head tilting and handheld camera angle variation. |
| **Translation ($\le 20\%$)** | **YES** | Enforces position invariance for off-center pets. |
| **RandomResizedCrop** | **YES** | Forces learning from partial body/face cues; simulates camera distance. |
| **Brightness Jitter** | **YES** | Simulates varied lighting (indoor vs outdoor sunlight). |
| **Contrast Jitter** | **YES** | Simulates varying lens/sensor quality and atmospheric haze. |
| **Saturation Jitter** | **YES** | Cat/dog breeds rely on pattern morphology, not ultra-exact saturation. |
| **Hue Jitter ($\le 0.05$)** | **MAYBE** | Mild shifts are fine; heavy shifts alter coat color (Golden Retriever $\rightarrow$ red). |
| **Gaussian Blur** | **YES** | Mitigates texture bias and simulates out-of-focus camera sensors. |
| **Perspective ($\le 0.3$)** | **MAYBE** | Mild camera angles are realistic; extreme homographies warp body shape. |
| **Random Erasing ($\le 20\%$)**| **YES** | Simulates partial occlusion by furniture, foliage, or other pets. |

---

## Exercise 26 — Create a Learnable Convolution

* **Data**: PyTorch Conv2d parameter inspection.
* **Program**: Instantiated `conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)`.
* **Observation**:
  * `conv.weight.shape` = `[8, 3, 3, 3]`
  * `conv.bias.shape` = `[8]`
  * Total parameters = $(8 \times 3 \times 3 \times 3) + 8 = 224$ weights.
* **What We Learned**: The 4D weight tensor dimensions represent $[C_{\text{out}}, C_{\text{in}}, K_H, K_W]$. Each output channel has its own distinct 3D filter that convolves across all input channels simultaneously.
* **Practical Use**: Fundamental building block of all convolutional backbones (ResNet, VGG, MobileNet).
* **Limitation**: Standard Conv2d has fixed kernel size and static receptive field.
* **Next Question**: What happens to spatial dimensions when a batch of images passes through this layer?

### Teacher's Evaluation Questions & Answers
1. *Explain the four numbers in the weight tensor shape `[8, 3, 3, 3]`.*
   **Answer**:
   * `8` = number of output filters / feature maps ($C_{\text{out}}$).
   * `3` = number of input channels ($C_{\text{in}}$, matching RGB).
   * `3` = spatial height of each filter kernel ($K_H$).
   * `3` = spatial width of each filter kernel ($K_W$).
2. *How many learnable parameters (including bias) exist in `nn.Conv2d(16, 32, kernel_size=5, bias=True)`?*
   **Answer**: Weights: $32 \times 16 \times 5 \times 5 = 12,800$. Biases: $32$. Total = $12,800 + 32 = 12,832$ parameters.

---

## Exercise 27 — Pass Data Through Convolution

* **Data**: Synthetic batch tensor $x \in \mathbb{R}^{1 \times 3 \times 32 \times 32}$.
* **Program**: Passed $x$ through `conv = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1)`.
* **Observation**: Output tensor $y$ has shape `[1, 8, 30, 30]`. Spatial dimensions shrank from $32 \times 32$ to $30 \times 30$.
* **What We Learned**: Without padding, a $K \times K$ kernel cannot calculate boundary outputs, losing $(K - 1)$ pixels along each spatial dimension according to:
  $$H_{\text{out}} = \lfloor \frac{H_{\text{in}} - K}{S} \rfloor + 1 = \frac{32 - 3}{1} + 1 = 30$$
* **Practical Use**: Feature map dimension calculation during manual network design.
* **Limitation**: Deep networks without padding rapidly shrink spatial maps to $0 \times 0$.
* **Next Question**: How can we configure padding to maintain exact spatial dimensions?

### Teacher's Evaluation Questions & Answers
1. *Why did the spatial dimensions shrink from $32 \times 32$ to $30 \times 30$ when passing through a $3\times3$ filter with `padding=0`?*
   **Answer**: A $3\times3$ kernel needs 1 pixel of context on all four sides. Without padding, the kernel center cannot visit the 1-pixel outer border of the input tensor, losing 2 pixels horizontally and 2 vertically ($32 - 2 = 30$).
2. *What is the formula for the output height $H_{\text{out}}$ of a 2D convolution?*
   **Answer**: $H_{\text{out}} = \lfloor \frac{H_{\text{in}} + 2P - K}{S} \rfloor + 1$, where $P$ is padding, $K$ is kernel size, and $S$ is stride.

---

## Exercise 28 — Padding

* **Data**: Synthetic tensor $x \in \mathbb{R}^{1 \times 3 \times 32 \times 32}$.
* **Program**: Executed `nn.Conv2d(3, 8, kernel_size=3, padding=1)`.
* **Observation**: Output tensor shape = `[1, 8, 32, 32]`. Spatial height and width are preserved.
* **What We Learned**: Setting $P = \frac{K - 1}{2}$ for odd $K$ (with stride $S=1$) produces "same" padding, keeping input and output spatial dimensions identical.
* **Practical Use**: Standard design pattern in deep architectures (e.g., VGG, ResNet) enabling networks of 50–100+ layers without spatial collapse.
* **Limitation**: Zero-padding introduces artificial 0-value boundaries at the edges of the image.
* **Next Question**: How do we intentionally downsample spatial feature maps using stride?

### Teacher's Evaluation Questions & Answers
1. *What padding value is required to keep spatial dimensions constant for a $5\times5$ convolution with stride 1?*
   **Answer**: $P = \frac{5 - 1}{2} = 2$. Padding 2 zeros on top, bottom, left, and right preserves the dimension ($32 + 4 - 5 + 1 = 32$).
2. *What is "reflection padding" and why is it sometimes used instead of zero padding?*
   **Answer**: Reflection padding mirrors border pixel values rather than padding with zeros, avoiding sharp artificial intensity discontinuities at image edges in tasks like super-resolution or style transfer.

---

## Exercise 29 — Stride

* **Data**: Synthetic tensor $x \in \mathbb{R}^{1 \times 3 \times 32 \times 32}$.
* **Program**: Tested `stride=1` vs `stride=2` with `padding=1`: `nn.Conv2d(3, 8, 3, stride=2, padding=1)`.
* **Observation**:
  * `stride=1` $\rightarrow [1, 8, 32, 32]$
  * `stride=2` $\rightarrow [1, 8, 16, 16]$ (spatial dimension halved).
* **What We Learned**: Stride controls the step size of the sliding kernel. Stride $>1$ performs strided convolution, downsampling spatial resolution and expanding effective receptive field.
* **Practical Use**: Strided convolutions replace pooling layers in modern CNNs (e.g., ResNet strided blocks).
* **Limitation**: Striding discards fine spatial information and can cause aliasing artifacts.
* **Next Question**: Why do linear convolutions require non-linear activations like ReLU between layers?

### Teacher's Evaluation Questions & Answers
1. *What is the primary architectural purpose of using `stride=2` in a convolutional layer?*
   **Answer**: It halves the spatial height and width ($4\times$ reduction in spatial area), reducing computational complexity and memory footprint in deeper layers while increasing the receptive field.
2. *How does strided convolution compare to MaxPooling for downsampling?*
   **Answer**: MaxPooling is a fixed, non-trainable operation that takes the maximum value, whereas strided convolution uses learnable kernel weights to downsample adaptively during backpropagation.

---

## Exercise 30 — ReLU Activation

* **Data**: 1D Tensor $x = [-5.0, -2.0, 0.0, 2.0, 8.0]$.
* **Program**: Applied `nn.ReLU()`: $y = \max(0, x)$.
* **Observation**: Negative inputs $[-5, -2]$ collapsed to $0.0$; positive inputs $[2, 8]$ remained untouched $[2.0, 8.0]$.
* **What We Learned**: Without non-linear activation functions, stacking multiple linear convolution layers is mathematically equivalent to a single linear layer ($W_2 W_1 x = W_{\text{combined}} x$). ReLU provides non-linearity with constant derivative ($1.0$) for positive values, preventing vanishing gradients.
* **Practical Use**: Default activation across virtually all computer vision deep learning models.
* **Limitation**: "Dying ReLU" problem: neurons with perpetually negative inputs have gradient $0$ and stop updating.
* **Next Question**: How do we pool activations over spatial neighborhoods to build translation invariance?

### Teacher's Evaluation Questions & Answers
1. *Why does stacking multiple convolutional layers without activation functions fail to create a deep network?*
   **Answer**: The composition of linear functions is always another linear function ($f(g(x)) = W_2(W_1 x) = (W_2 W_1)x = W_3 x$). Without non-linear activations, a 100-layer network has no more representational power than a 1-layer linear model.
2. *What is the "Dying ReLU" problem and what activation variants address it?*
   **Answer**: If a large gradient updates a neuron such that it outputs negative values for all dataset inputs, its gradient becomes permanently 0 and the neuron never recovers. Variants like LeakyReLU ($y = \max(\alpha x, x)$ with $\alpha=0.01$), PReLU, and GELU address this.

---

## Exercise 31 — Max Pooling

* **Data**: Structured $4 \times 4$ tensor with known regional peak values.
* **Program**: Applied `nn.MaxPool2d(kernel_size=2, stride=2)`.
* **Observation**: $4 \times 4$ spatial matrix reduced to $2 \times 2$; each $2 \times 2$ quadrant was replaced by its local maximum.
* **What We Learned**: Max pooling extracts the most prominent feature activation in each sub-region, providing local translation invariance and halving spatial dimensions.
* **Practical Use**: Downsampling feature maps in classical CNNs (LeNet, AlexNet, VGG).
* **Limitation**: Discards exact spatial coordinates of activations within the pool window.
* **Next Question**: How do we combine Conv, ReLU, Pool, and Linear layers into an end-to-end classification network?

### Teacher's Evaluation Questions & Answers
1. *What information is retained and what is lost during a MaxPool2d operation?*
   **Answer**:
   * **Retained**: The presence and maximum strength of a detected feature within that region.
   * **Lost**: The exact spatial $(x, y)$ coordinate of where that feature occurred within the pooling window.
2. *Does `nn.MaxPool2d(2)` have any learnable parameters?*
   **Answer**: No. Max pooling is a fixed, parameter-free deterministic operation with zero weights and zero biases.

---

## Exercise 32 — Build a Tiny CNN

* **Data**: Synthetic CIFAR-10 tensor ($1 \times 3 \times 32 \times 32$).
* **Program**: Implemented complete PyTorch `nn.Sequential` pipeline:
  ```python
  model = nn.Sequential(
      nn.Conv2d(3, 16, 3, padding=1),   # (1, 16, 32, 32)
      nn.ReLU(),
      nn.MaxPool2d(2),                  # (1, 16, 16, 16)
      nn.Conv2d(16, 32, 3, padding=1),  # (1, 32, 16, 16)
      nn.ReLU(),
      nn.MaxPool2d(2),                  # (1, 32, 8, 8)
      nn.Flatten(),                     # (1, 2048)
      nn.Linear(32 * 8 * 8, 10)         # (1, 10)
  )
  ```
* **Observation**: Traced shape flow from $[1, 3, 32, 32]$ down to 10 class logits. Total parameters = $25,642$.
* **What We Learned**: A CNN progressively expands channel capacity ($3 \rightarrow 16 \rightarrow 32$) while contracting spatial dimensions ($32 \rightarrow 16 \rightarrow 8$), and finally maps spatial feature representations to class logits using a fully connected layer.
* **Practical Use**: Lightweight embedded vision classification, baseline image classifiers.
* **Limitation**: Lacks Batch Normalization and Dropout; fully connected head is sensitive to spatial dimensions.
* **Next Question**: How do we convert the raw 10-dimensional output logits into valid class probabilities?

### Complete Shape Trace Table
| Layer Index | Layer Type | Output Tensor Shape | Parameter Count |
| :---: | :--- | :--- | :--- |
| $0$ | `Conv2d(3, 16, kernel_size=3, pad=1)` | `[1, 16, 32, 32]` | $(16 \times 3 \times 3 \times 3) + 16 = 448$ |
| $1$ | `ReLU()` | `[1, 16, 32, 32]` | $0$ |
| $2$ | `MaxPool2d(kernel_size=2, stride=2)` | `[1, 16, 16, 16]` | $0$ |
| $3$ | `Conv2d(16, 32, kernel_size=3, pad=1)`| `[1, 32, 16, 16]` | $(32 \times 16 \times 3 \times 3) + 32 = 4,640$ |
| $4$ | `ReLU()` | `[1, 32, 16, 16]` | $0$ |
| $5$ | `MaxPool2d(kernel_size=2, stride=2)` | `[1, 32, 8, 8]` | $0$ |
| $6$ | `Flatten()` | `[1, 2048]` | $0$ |
| $7$ | `Linear(in_features=2048, out=10)` | `[1, 10]` | $(2048 \times 10) + 10 = 20,490$ |
| **Total**| — | — | **25,578 weights + 64 biases = 25,642** |

---

## Exercise 33 — Classification Output (Logits $\rightarrow$ Softmax)

* **Data**: Synthetic CIFAR-10 image tensor passed through Tiny CNN from Exercise 32.
* **Program**: Extracted unnormalized raw logits, applied Softmax:
  $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
  Identified predicted class via `torch.argmax()`.
* **Observation**: Raw logits contain negative and unconstrained numbers. Softmax maps logits to positive values in range $(0, 1)$ that sum to $1.000000$.
* **What We Learned**: Softmax transforms unconstrained scores into a valid multinomial probability distribution over $C$ mutually exclusive classes.
* **Practical Use**: Final decision output of all multi-class classifiers.
* **Limitation**: Softmax can be overconfident on out-of-distribution inputs (allocates high probability to arbitrary classes).
* **Next Question**: Why is raw accuracy a dangerous and misleading metric on imbalanced real-world datasets?

### Teacher's Evaluation Questions & Answers
1. *Why are raw logits from a linear layer not considered probabilities?*
   **Answer**: Raw logits can be negative, greater than 1, and do not sum to 1.0, violating the fundamental axioms of probability ($0 \le P(E) \le 1$ and $\sum P = 1$).
2. *Why is Cross-Entropy loss in PyTorch (`nn.CrossEntropyLoss`) computed directly from logits rather than from softmax outputs?*
   **Answer**: Combining LogSoftmax and Negative Log-Likelihood into a single mathematical step uses the "log-sum-exp trick," which provides numerical stability and avoids arithmetic underflow/overflow from extreme exponential values.

---

## Exercise 34 — Accuracy Can Mislead (Imbalance Case Study)

* **Data**: Industrial inspection dataset: $950$ normal units, $50$ defective units (5% defect rate).
* **Program**: Evaluated a "lazy" model that blindly predicts "NORMAL" for every single image.
* **Observation**:
  * **Accuracy**: $\frac{950 + 0}{1000} = \mathbf{95.0\%}$ (appears high).
  * **True Positives (TP)**: $0$ (caught 0 defects).
  * **False Negatives (FN)**: $50$ (every single defect was missed).
  * **Defect Recall**: $\frac{0}{0 + 50} = \mathbf{0.0\%}$ (complete system failure).
  * **Defect Precision**: Undefined / $0.0\%$.
* **What We Learned**: High accuracy on imbalanced datasets is misleading. A model with 95% accuracy can have a 0% recall rate on the class that matters most.
* **Practical Use**: Quality control, medical screening (cancer vs benign), fraud detection.
* **Limitation**: Precision and Recall individually can still be traded off; F1-score or cost-weighted loss is needed for balance.
* **Next Question**: How can we display all classification outcomes (TP, FP, TN, FN) simultaneously?

### Teacher's Evaluation Questions & Answers
1. *Why is 95% accuracy unacceptable for this factory defect detection system?*
   **Answer**: Because the system has **0% recall** on defective items. All 50 defective products bypass inspection and ship to customers, risking equipment damage, safety violations, and product recalls.
2. *Define Precision and Recall in the context of defect detection.*
   **Answer**:
   * **Precision**: Of all items the model flagged as defective, what percentage was actually defective? ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
   * **Recall**: Of all actual defective items produced, what percentage did the model successfully catch? ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).

---

## Exercise 35 — Confusion Matrix

* **Data**: Ground truth `y_true = [1, 1, 1, 0, 0, 0]`, predictions `y_pred = [1, 0, 1, 0, 0, 1]`.
* **Program**: Computed confusion matrix using `sklearn.metrics.confusion_matrix`:
  $$\text{CM} = \begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$
  Visualized annotated 2D heatmap with per-sample tracking.
* **Observation**:
  * $\text{TN} = 2$ (correct negative)
  * $\text{FP} = 1$ (false alarm / Type I error)
  * $\text{FN} = 1$ (missed positive / Type II error)
  * $\text{TP} = 2$ (correct positive)
* **What We Learned**: The confusion matrix gives a complete breakdown of errors, showing whether a model struggles with false alarms (low precision) or missed detections (low recall).
* **Practical Use**: Comprehensive model diagnostic for multi-class classification and error auditing.
* **Limitation**: With hundreds of classes, raw counts become hard to interpret without row-wise normalization.
* **Next Question**: How did the historical evolution of CNN architectures solve foundational engineering bottlenecks?

### Teacher's Evaluation Questions & Answers
1. *What is the difference between a Type I error and a Type II error in a confusion matrix?*
   **Answer**:
   * **Type I error (False Positive)**: Model falsely predicts positive for a true negative sample (false alarm).
   * **Type II error (False Negative)**: Model falsely predicts negative for a true positive sample (missed detection).
2. *How is the multi-class confusion matrix organized?*
   **Answer**: Rows represent ground-truth actual classes, columns represent model predicted classes. True predictions lie along the main diagonal; off-diagonal elements indicate specific misclassifications between pairs of classes.

---

## Exercise 36 — CNN Architecture Thought Experiment

### Evolutionary Architecture Analysis

#### 1. LeNet-5 (1998 — Yann LeCun et al.)
* **Engineering Problem**: How to classify handwritten check digits (MNIST) without brittle, hand-crafted feature engineering.
* **Core Breakthrough**: Established the modern blueprint of computer vision: interleaved 2D convolutions, spatial subsampling (average pooling), and fully connected output layers trained end-to-end via backpropagation.

#### 2. AlexNet (2012 — Alex Krizhevsky et al.)
* **Engineering Problem**: Training deep multi-layer CNNs on 1.2 million high-resolution ImageNet images without exploding/vanishing gradients or massive overfitting.
* **Core Breakthroughs**:
  1. **GPU Acceleration**: Implemented fast custom CUDA convolution kernels across 2 parallel GPUs.
  2. **ReLU Activation**: Replaced saturating sigmoid/tanh with non-saturating ReLU, speeding up convergence by $6\times$.
  3. **Dropout Regularization**: Injected 50% dropout in dense layers to prevent complex co-adaptations.

#### 3. VGG-16 / VGG-19 (2014 — Simonyan & Zisserman)
* **Engineering Problem**: How to increase network depth without introducing arbitrary, ad-hoc filter geometry at each layer.
* **Core Breakthrough**: Standardized the entire architecture to use only small $3 \times 3$ convolution filters stacked consecutively.
  * **Why $3 \times 3$?** Two consecutive $3 \times 3$ convolutions have an effective receptive field of $5 \times 5$, but use $(2 \times 3^2 \times C^2) = 18 C^2$ parameters instead of $25 C^2$ ($28\%$ parameter reduction), while introducing two non-linear ReLU activations instead of one.

#### 4. GoogLeNet / Inception (2014 — Szegedy et al.)
* **Engineering Problem**: How to capture both fine local textures and large global structures without blowing up computational and memory budgets.
* **Core Breakthrough**: Multi-scale "Inception Modules" running $1 \times 1$, $3 \times 3$, and $5 \times 5$ convolutions in parallel at the same depth. Used $1 \times 1$ "bottleneck" convolutions to project channels into lower dimensions before expensive large convolutions, enabling 22-layer depth with only 5 million parameters ($12\times$ fewer than AlexNet).

#### 5. ResNet (2015 — Kaiming He et al.)
* **Engineering Problem**: The **Degradation Problem** — stacking more layers caused training error (not just test error) to worsen because gradients vanished or degraded when backpropagating through 50+ non-linear layers.
* **Core Breakthrough & Formula**:
  $$y = \mathcal{F}(x, \{W_i\}) + x$$
  * **Plain Language Explanation**: Instead of forcing the layer stack $\mathcal{F}(x)$ to learn the complete mapping from input $x$ to output $y$, the network only needs to learn the **residual difference** (the modification or adjustment) $\mathcal{F}(x) = y - x$.
  * If no modification is needed, the optimizer easily sets the weights in $\mathcal{F}(x)$ toward zero, and the block acts as a transparent identity pass-through ($y = x$).
  * During backpropagation, the derivative $\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + 1$ always carries a constant $+1$ term, allowing error gradients to flow straight back to early layers without vanishing.

---

# PART 2: FINAL PROJECT EVALUATION

*The following section contains complete, rigorous answers to the Teacher's Final Project Evaluation criteria based on the practical computer vision system implemented across Exercises 1–36.*

### 1. Data Ingestion & Source
* **Visual Data Consumed**: Single-frame 2D RGB planar images ($H \times W \times 3$) and single-channel grayscale maps ($H \times W$).
* **Data Sources**: Oxford-IIIT Pet benchmark images and synthetic test patterns that evaluate geometric, photometric, and boundary responses.
* **Available Samples**: 7,349 images across 37 cat and dog breeds (Oxford Pets), supplemented by synthetic geometric test cases.
* **Labels**: Categorical discrete class indices ($0 \dots 36$) representing biological breeds, accompanied by ground-truth binary masks for segmentation.
* **Task Type**: Multi-class visual classification paired with localized semantic edge/color mask detection.
* **Expected Output**: 
  1. $C$-dimensional probability vector from Softmax indicating the most likely class index ($\hat{y} = \arg\max P(Y=c|X)$).
  2. Binary spatial segmentation mask ($M \in \{0, 1\}^{H \times W}$) and bounding boxes $[x, y, w, h]$.

### 2. Classical Computer Vision vs Deep Learning Trade-offs
* **Can Classical CV Solve Parts of the Problem?**
  * **Yes**. Bounding region extraction, foreground thresholding (HSV), noise denoising (Gaussian/Bilateral blur), and contrast equalization can be handled faster and more predictably by classical algorithms than by deep networks.
* **Can Deterministic Rules Replace ML?**
  * For constrained environments (e.g., detecting bright green markers under controlled LED illumination), deterministic HSV rules (`cv2.inRange`) replace ML completely with zero GPU requirements, sub-millisecond latency, and 100% interpretability.
  * For unconstrained pet breed classification, deterministic rules fail because biological intra-class variance and background clutter cannot be captured by manual if-else statements.

### 3. Real-World Environmental Invariance Analysis
* **Rotation**: Small angular variations ($\pm 15^\circ$) are handled by data augmentation and CNN pooling; full $360^\circ$ rotation requires explicit rotational augmentation or spatial transformer networks (STNs).
* **Position**: Handled by convolutional weight sharing (translation equivariance) and spatial max/average pooling.
* **Scale**: Handled by multi-scale training (`RandomResizedCrop`), feature pyramid networks (FPN), and hierarchical receptive field expansion across deep layers.
* **Lighting & Color**: Handled by photometric data augmentation (`ColorJitter`), HSV decoupling, and Batch Normalization layers that standardize channel activations.
* **Perspective & Viewpoint**: Addressed through random homography augmentation (`RandomPerspective`) and multi-view training sets.
* **Occlusion**: Addressed via CutOut / `RandomErasing` transformations during training, forcing the network to rely on multiple non-local features.
* **Image Quality / Blur**: Addressed by training with Gaussian blur and random downsampling to prevent texture over-reliance.

### 4. Layer Representations & Feature Hierarchy
* **Early CNN Layers (Layers 1–2)**: Learn low-level, generic Gabor-like filters — directional edges (Sobel-like), color gradients, corners, and simple textural frequency patches.
* **Deep CNN Layers (Layers 4+)**: Combine intermediate motifs into complex, class-specific semantic representations (e.g., snout geometry, ear shape, eye spacing).

### 5. Metric Selection & Cost of Errors
* **Cost of Errors**: In high-stakes detection (e.g., quality control defect screening or veterinary triage), False Negatives (missing a defect/pathology) carry much higher cost than False Positives (manual reinspection).
* **Accuracy vs Precision/Recall**: When defect/target rates are small ($<5\%$), accuracy is uninformative (95% accuracy with 0% recall). We prioritize **Recall** and **$F_{\beta}$-score** ($\beta=2$) to penalize missed positive cases.

### 6. Validation Protocol & Data Leakage Prevention
* **Data Split Protocol**: Strict 70% Train, 15% Validation, 15% Test split partitioned at the *subject / scene level* (not random frame level) to prevent identical pets/backgrounds appearing in both train and test splits.
* **Data Leakage Safeguards**: All data augmentation and normalization statistics ($\mu, \sigma$) are computed **strictly on the training fold** and applied frozen to validation and test folds.

### 7. Deployment, Latency & Failure Recovery
* **Deployment Environment**: Target edge inference on CPU / embedded platforms (Raspberry Pi, Apple Silicon, ONNX Runtime).
* **Real-Time Feasibility**: The Tiny CNN (25K parameters) executes in $<2.5\text{ ms}$ on standard CPU, exceeding 60 FPS real-time requirements.
* **Out-of-Distribution (OOD) Handling**: Compute entropy of the Softmax distribution; if prediction entropy exceeds threshold $H(p) > \tau$, flag the sample as OOD / Uncertain and trigger manual review.
* **Post-Deployment Monitoring**: Log prediction confidence distributions and periodically sample production inferences for human auditing and drift detection.

---

# PART 3: RAPID ORAL EVALUATION

### Question 1: What does a kernel actually do?
**Answer**:
A kernel is a small numerical matrix (e.g., $3 \times 3$) that slides across an input image/tensor. At each position, it computes an element-wise multiplication with the overlapping patch of pixels followed by a summation (a local dot product or 2D cross-correlation). Depending on its values, it acts as a spatial filter that computes derivatives (edge detection), weighted averages (smoothing/blurring), or learned pattern matching (CNN feature detection).

---

### Question 2: Why is augmentation not the same as collecting new data?
**Answer**:
Data augmentation applies mathematical transformations (rotations, crops, color shifts) to *existing* images. While it expands the geometric and photometric variety of the training distribution and regularizes network weights, it **cannot create new semantic information, new classes, novel backgrounds, or unobserved biological variations** that were not present in the original dataset.

---

### Question 3: Name one augmentation that can destroy labels and explain why.
**Answer**:
**Vertical Flip** (or extreme **Hue Jitter**). 
* In digit recognition (MNIST), a vertical flip turns a '6' into a '9' or an upside-down '7' into an invalid symbol.
* In traffic signal detection, Hue Jitter can shift a red light to green, inverting the ground-truth safety label.
* In natural animal photos, vertical flip places the ground on top and animals on the ceiling, contradicting natural physical priors.

---

### Question 4: Why might a deterministic CV solution be preferable to a CNN?
**Answer**:
1. **Zero Training & No Data Requirement**: Does not require thousands of annotated training images or GPU hardware.
2. **Deterministic & Interpretable**: Predictable mathematical behavior with 100% auditable failure modes.
3. **Ultra-Low Latency & Low Compute**: Executes in microseconds on low-power microcontrollers without neural accelerators.
4. **Guaranteed Corner Case Constraints**: Hard mathematical thresholds cannot be "fooled" by out-of-distribution adversarial pixel noise.

---

### Question 5: What is the difference between a manually designed Sobel kernel and `nn.Conv2d`?
**Answer**:
* A **Sobel kernel** has fixed, hand-crafted analytical coefficients $\begin{bmatrix}-1&0&+1\\-2&0&+2\\-1&0&+1\end{bmatrix}$ specifically derived to approximate spatial image derivatives $\frac{\partial I}{\partial x}$. Its weights are static and never change.
* An **`nn.Conv2d`** layer initializes its kernel weights randomly and uses backpropagation and stochastic gradient descent to **automatically learn** optimal filtering coefficients from data. It can discover edge detectors, texture filters, or complex combinations tailored to minimizing task loss.
