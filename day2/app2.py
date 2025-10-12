import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧩 Image Preprocessing Playground")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = np.array(Image.open(uploaded_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.subheader("Original and Grayscale")
    st.image([image, gray], caption=["Original Image", "Grayscale Image"], use_container_width=True)

    # Operation selection
    operation = st.selectbox(
        "Select Operation",
        ("Blurring / Smoothing", "Edge Detection", "Morphological Operations")
    )

    # ---------------------------------
    # 1️⃣ BLURRING / SMOOTHING
    # ---------------------------------
    if operation == "Blurring / Smoothing":
        blur_type = st.radio("Choose blur type", ("Average Blur", "Gaussian Blur", "Median Blur"))
        k = st.slider("Kernel Size", 1, 25, 5, step=2)

        if blur_type == "Average Blur":
            processed = cv2.blur(image, (k, k))
        elif blur_type == "Gaussian Blur":
            processed = cv2.GaussianBlur(image, (k, k), 0)
        else:
            processed = cv2.medianBlur(image, k)

        st.image(processed, caption=f"{blur_type} (k={k})", use_container_width=True)

    # ---------------------------------
    # 2️⃣ EDGE DETECTION
    # ---------------------------------
    elif operation == "Edge Detection":
        low = st.slider("Lower Threshold", 0, 255, 50)
        high = st.slider("Upper Threshold", 0, 255, 150)
        edges = cv2.Canny(gray, low, high)
        st.image(edges, caption="Canny Edge Detection", use_container_width=True)

    # ---------------------------------
    # 3️⃣ MORPHOLOGICAL OPERATIONS
    # ---------------------------------
    elif operation == "Morphological Operations":
        morph_type = st.selectbox(
            "Choose operation", 
            ("Erosion", "Dilation", "Opening", "Closing")
        )
        k = st.slider("Kernel Size", 1, 25, 5, step=2)
        kernel = np.ones((k, k), np.uint8)

        if morph_type == "Erosion":
            processed = cv2.erode(gray, kernel, iterations=1)
        elif morph_type == "Dilation":
            processed = cv2.dilate(gray, kernel, iterations=1)
        elif morph_type == "Opening":
            processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        else:
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        st.image(processed, caption=f"{morph_type} (Kernel={k}x{k})", use_container_width=True)
