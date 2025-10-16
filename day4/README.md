# Day 4: Histogram of Oriented Gradients (HOG)

Understanding feature extraction before the deep learning era.

## Overview

Histogram of Oriented Gradients (HOG) is a feature descriptor introduced by Dalal and Triggs (2005) for object detection. It was the state-of-the-art method for pedestrian detection from 2005-2012, before deep learning dominance.

## Key Concepts

**HOG Pipeline:**
1. Divide image into small cells (8x8 pixels)
2. Compute gradient magnitude and orientation for each pixel
3. Create histogram of gradient orientations for each cell (9 bins, 0-180 degrees)
4. Normalize histograms over larger blocks (2x2 cells) using L2-norm
5. Concatenate all normalized histograms into feature vector

**Feature Dimensionality:**
For 64x128 image with standard parameters:
- Cells: 8 x 16 = 128 cells
- Blocks: 7 x 15 = 105 blocks (with overlap)
- Features per block: 2x2 cells x 9 orientations = 36
- Total features: 105 x 36 = 3780 dimensions

## Folder Structure

```
day4/
├── HoG.ipynb                  # Main notebook
├── images/
│   ├── pedestrian.jpg         # Sample pedestrian image
│   ├── non-pedestrian.png     # Sample non-pedestrian image
│   ├── pedestrians/           # Training data (614 images)
│   └── non_pedestrians/       # Training data (600 images)
└── README.md
```

## Requirements

```bash
pip install opencv-python scikit-image scikit-learn matplotlib numpy
```

## Notebook Contents

**Section 1: Setup and Imports**
- Import required libraries (cv2, numpy, matplotlib, sklearn, skimage)

**Section 2: What is HOG?**
- Theory and background
- Importance before CNNs

**Section 3: Image Gradients**
- Gradient magnitude and orientation computation
- Sobel operators for derivative approximation
- Visualization of gradients

**Section 4: HOG Descriptor Formation**
- Pipeline overview
- Mathematical formulation (L2-norm)
- Feature extraction with scikit-image
- HOG visualization

**Section 5: Manual HOG Computation**
- Custom implementation for educational purposes
- Histogram visualization for sample cells
- Cell grid overlay visualization

**Section 6: HOG + SVM for Pedestrian Detection**
- Dataset loading using glob
- Feature extraction (3780 features per 64x128 image)
- SVM classifier training (linear kernel)
- Model evaluation (classification report)
- Individual image prediction
- OpenCV pre-trained HOG detector

**Section 7: Parameter Comparison**
- Effect of different orientations (9 vs 12)
- Effect of cell size (8x8 vs 16x16)
- Effect of block size (2x2 vs 3x3)

**Section 8: Conclusion**
- Advantages and limitations
- Historical impact

## Dataset

The notebook uses the INRIA Person Dataset:
- Positive samples: 614 pedestrian images
- Negative samples: 600 non-pedestrian images
- All resized to 64x128 pixels for training

## Results

Training a linear SVM on HOG features achieves approximately 84% accuracy on the test set for pedestrian vs non-pedestrian classification.

## Key Parameters

```python
orientations = 9              # Number of orientation bins
pixels_per_cell = (8, 8)     # Cell size in pixels
cells_per_block = (2, 2)     # Block size in cells
```

## Usage

1. Ensure images are in the `images/` directory
2. Open `HoG.ipynb` in Jupyter Notebook or VS Code
3. Run cells sequentially

## Historical Context

**Why HOG mattered:**
- Robust to geometric and photometric transformations
- Efficient computation compared to raw pixels
- State-of-the-art for pedestrian detection (2005-2012)
- Foundation for understanding feature descriptors

**Limitations:**
- Fixed feature representation
- Manual parameter tuning required
- Outperformed by deep learning on large datasets
