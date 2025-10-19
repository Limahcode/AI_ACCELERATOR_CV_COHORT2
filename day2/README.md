# Day 2 
Welcome to day 2! 


# 🖼️ Image Processing Playground (Streamlit Mini-Project)

This mini-project is designed to help you apply various image processing techniques interactively using a **Streamlit web app**. You'll build an app that lets users upload images, apply transformations, and view both the original and processed images side by side.

---

## 🚀 Project Goals

Build a Streamlit App that:

- Allows users to upload images.
- Lets users select and apply a variety of image processing techniques covered in class.
- Visualizes before-and-after results for intuitive understanding.

Reinforce your learning by experimenting with:

- Thresholding  
- Smoothing and Blurring  
- Edge Detection  
- Contour Detection  
- Template Matching  
- Image Segmentation (Watershed)  
- Color Space Transformations

---

## 🛠️ Features & Techniques

The app allows users to apply any of the following techniques:

### ✅ Thresholding

- **Simple Thresholding**: Binary, Inverted, Truncation, ToZero, etc.
- **Adaptive Thresholding**: Region-based thresholding for varying lighting.
- **Otsu's Thresholding**: Automatically determines the optimal threshold.

### ✅ Blurring & Smoothing

- **Averaging Blur**: Removes noise by averaging pixels.
- **Gaussian Blur**: Weighted blur that preserves edges better.
- **Median Blur**: Best for removing salt-and-pepper noise.
- **2D Convolution**: Apply custom kernels using `cv.filter2D`.

### ✅ Edge Detection

- **Sobel / Scharr Filters**: Compute gradients in X or Y direction.
- **Laplacian Filter**: Second-order derivative for edge detection.
- **Canny Edge Detection**: Multi-stage, accurate edge detection.

### ✅ Contour Detection

- Detect and visualize object boundaries.
- Draw detected contours on the image.

### ✅ Template Matching

- Match smaller template image inside a larger one.
- Supports six OpenCV template matching methods.

### ✅ Watershed Segmentation

- Segment overlapping objects (e.g., touching coins).
- Visualize boundaries using marker-based flooding.

### ✅ Color Space Conversion

- Convert between BGR, RGB, Grayscale, and HSV.
- Great for preprocessing and filtering.

### ✅ Image Operations

- Resize, crop, mask, and change brightness/contrast.
- Overlay cropped regions onto other parts of an image.

---


---