# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Introduction & Motivation

**Image Segmentation** is the process of partitioning a digital image into multiple meaningful segments or pixel classes. Unlike simple image classification (which assigns a single label to an entire image), segmentation performs **pixel-level classification**, generating a spatial mask that pinpoints both the identity, boundary, and exact location of target objects.

* **Key Applications:**
  * **Satellite & Remote Sensing:** Automatically segmenting roads, lakes, buildings, land cover, and green vegetation from aerial imagery.
  * **Medical Imaging:** Identifying and delineating tumors, lesions, and anatomical structures in X-ray, Ultrasound, MRI, and CT scans to assist clinical diagnosis.
* **Mask / Ground Truth Representation:**
  * **Binary Segmentation:** Single foreground target (e.g., tumor $= 1$) against background ($= 0$).
  * **Multi-Class Segmentation:** Multiple distinct object types represented by unique integer IDs ($0 = \text{background}, 1 = \text{roads}, 2 = \text{buildings}, 3 = \text{water}$).

```
┌──────────────────────────────────────────────────────────────┐
│                    Image Segmentation Flow                   │
│                                                              │
│   Input Image                 Trained U-Net       Output Mask│
│   ┌──────────────┐          ┌───────────────┐     ┌────────┐ │
│   │  Spectrogram │  ──────> │ Encoder-      │ ──> │ Pixel  │ │
│   │  or Medical  │          │ Decoder with  │     │ Binary │ │
│   │  Ultrasound  │          │ Skip Conns    │     │ Mask   │ │
│   └──────────────┘          └───────────────┘     └────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Mathematical Step-by-Step Walkthrough (5×5 Toy Example)

To understand how U-Net processes data under the hood, consider a single-channel normalized image matrix ($5 \times 5$ pixels with values in $[0, 1]$):

### 1. Convolution & Feature Maps
* **Kernel / Filter:** A small matrix of trainable weights (e.g., $2 \times 2$ or $3 \times 3$) plus a trainable bias term $b$.
* **Operation:** Compute element-wise dot product between the kernel and overlapping image patch, add bias $b$, and store in the feature map:
  $$\text{Output} = \sum_{i, j} (W_{i,j} \cdot X_{i,j}) + b$$
* **Stride:** Number of pixel steps the filter shifts per calculation ($\text{stride} = 1$).
* **Padding:** Adding zero-value borders so output feature maps retain the original spatial dimensions.

### 2. Activation Function (ReLU)
* **Rectified Linear Unit:** Applies non-linearity by clamping all negative values to zero:
  $$f(z) = \max(0, z)$$
* Ensures positive feature activations pass through unchanged while suppressing negative noise.

### 3. Downsampling (Max Pooling)
* **Operation:** Slides a $2 \times 2$ window with $\text{stride} = 2$, extracting only the maximum value within each window.
* **Effect:** Halves the spatial resolution ($H/2, W/2$), reducing computational complexity while expanding the receptive field to capture broader contextual features.

### 4. Upsampling Methods
* **Method A — Nearest Neighbor Upsampling:**
  * Duplicates each scalar value into adjacent $2 \times 2$ spatial cells.
  * Fast and non-parametric (contains no trainable weights).
* **Method B — Transposed Convolution (Deconvolution):**
  * Uses trainable filter weights and bias terms to project low-resolution feature maps back into higher-dimensional space.
  * Allows the network to learn optimal interpolation strategies dynamically.

### 5. Skip Connections & Concatenation
* **Mechanism:** Direct lateral data bridges that transfer high-resolution feature maps from the contracting encoder directly into the expanding decoder.
* **Why Essential:** Max pooling discards fine-grained spatial coordinates. Skip connections restore localized edge and boundary details lost during downsampling.

### 6. Final 1×1 Convolution & Probability Thresholding
* **$1 \times 1$ Convolution:** Projects multichannel feature maps into the desired number of class channels.
* **Sigmoid Activation (Binary Segmentation):**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Thresholding:** Classifies each pixel into foreground or background:
  $$\text{Class}(x, y) = \begin{cases} 1 & \text{if } P(x, y) \ge 0.5 \\ 0 & \text{if } P(x, y) < 0.5 \end{cases}$$

---

## Original U-Net Architecture (Ronneberger et al. 2015)

The U-Net architecture forms a symmetric **U-shape** divided into an **Encoder (Contracting Path)** and a **Decoder (Expansive Path)**:

```
Input (572x572x1) ──┐                                         ┌──> Output (388x388x2)
                    │                                         │
               [2x Conv 3x3] ────── Skip Connection ────> [2x Conv 3x3]
               (64 ch, 568x568)                           (64 ch, 388x388)
                    │                                         ▲
               [Max Pool 2x2]                           [Up-Conv 2x2]
                    │                                         │
               [2x Conv 3x3] ────── Skip Connection ────> [2x Conv 3x3]
               (128 ch, 280x280)                          (128 ch, 392x392)
                    │                                         ▲
               [Max Pool 2x2]                           [Up-Conv 2x2]
                    │                                         │
               [2x Conv 3x3] ────── Skip Connection ────> [2x Conv 3x3]
               (256 ch, 136x136)                          (256 ch, 200x200)
                    │                                         ▲
               [Max Pool 2x2]                           [Up-Conv 2x2]
                    │                                         │
               [2x Conv 3x3] ────── Skip Connection ────> [2x Conv 3x3]
               (512 ch, 64x64)                            (512 ch, 104x104)
                    │                                         ▲
               [Max Pool 2x2]                           [Up-Conv 2x2]
                    └──────────────> [2x Conv 3x3] ───────────┘
                                   Bottleneck (1024 ch, 28x28)
```

### Architectural Breakdown

1. **Contracting Path (Encoder):**
   * Captures semantic context and feature hierarchy.
   * Consists of repeated blocks: two $3 \times 3$ unpadded convolutions $\rightarrow$ ReLU $\rightarrow$ $2 \times 2$ Max Pooling ($\text{stride} = 2$).
   * Channel depth doubles at each stage: $64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$.

2. **Bottleneck:**
   * Deepest layer ($28 \times 28 \times 1024$), capturing the most abstract contextual representations.

3. **Expansive Path (Decoder):**
   * Enables precise spatial localization.
   * Consists of $2 \times 2$ Transposed Convolutions (halving channel depth, doubling spatial dimensions) $\rightarrow$ Concatenation with cropped encoder skip features $\rightarrow$ two $3 \times 3$ convolutions $\rightarrow$ ReLU.

4. **Output Head:**
   * $1 \times 1$ convolution mapping 64 channels to $N$ class logits (Softmax for multi-class, Sigmoid for binary).
   * **Total Parameters:** Approximately **31.03 Million trainable parameters**.

---

## Segmentation Evaluation Metrics

Evaluating segmentation models requires metrics tailored for spatial overlap rather than raw pixel accuracy:

### 1. Pixel Accuracy (Misleading for Sparse Targets)
$$\text{Pixel Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Limitation:** In medical/astronomical images where the target object occupies only $5\%$ of the image, a dumb model predicting all-background achieves **$95\%$ accuracy** while completely missing the target.

### 2. Intersection over Union (IoU / Jaccard Index)
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$
* Quantifies the exact ratio of overlapping target area against the union of predicted and ground-truth masks ($1.0 = \text{perfect match}$).

### 3. Dice Coefficient (F1 Score)
$$\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
* Heavily penalizes false positives and false negatives while emphasizing true foreground overlap.

---

## Python / Keras Implementation Guide (Breast Ultrasound Dataset)

### 1. Data Preprocessing & Mask Merging
```python
import os
import cv2
import numpy as np

def load_and_preprocess(image_paths, img_size=128):
    images, masks = [], []
    current_mask = None
    
    for path in sorted(image_paths):
        # Load and resize image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size)) / 255.0  # Min-max normalization
        
        if "mask" not in path:
            images.append(img)
            current_mask = np.zeros((img_size, img_size), dtype=np.float32)
        else:
            # Merge multi-tumor masks via logical OR / summation
            mask_data = (img > 0.5).astype(np.float32)
            current_mask = np.clip(current_mask + mask_data, 0, 1)
            
    return np.expand_dims(np.array(images), -1), np.expand_dims(np.array(masks), -1)
```

### 2. Building U-Net with Same Padding
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)
    
    # Encoder Block 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Encoder Block 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    
    # Decoder Block 2
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
    
    # Decoder Block 1
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    
    # Output Layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    return models.Model(inputs=[inputs], outputs=[outputs])
```

---

## Critical Training Techniques & Hyperparameter Optimizations

1. **He Normal (Kaiming) Weight Initialization:**
   * Weights are initialized from a Gaussian distribution:
     $$\mathcal{N}\left(0, \sigma = \sqrt{\frac{2}{n_{in}}}\right), \quad n_{in} = k_w \times k_h \times c_{in}$$
   * **Experimental Impact:** Boosted test IoU score from **$\sim 0.60$ to $\sim 0.70$** across 40 epochs on ultrasound segmentation.

2. **Loss Function Selection:**
   * **Binary Cross-Entropy (BCE):** Stable pixel-level gradients.
   * **Dice Loss / IoU Loss:** Directly optimizes foreground overlap and mitigates heavy background class imbalance.
   * **Hybrid Loss:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$.

3. **Modern Architecture Enhancements:**
   * **Residual Connections (Res-UNet):** Replaces standard conv blocks with residual skip additions to avoid vanishing gradients.
   * **Attention Gates (Attention U-Net):** Dynamically filters encoder skip connections to focus only on salient target regions.
