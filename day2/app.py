import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧠 Image Thresholding Playground")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image to OpenCV format
    image = np.array(Image.open(uploaded_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", use_container_width=True)

    # Select thresholding method
    method = st.selectbox(
        "Select Thresholding Method",
        ("Simple Thresholding", "Adaptive Thresholding", "Otsu's Thresholding")
    )

    if method == "Simple Thresholding":
        t = st.slider("Threshold Value", 0, 255, 127)
        _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    elif method == "Adaptive Thresholding":
        block_size = st.slider("Block Size (odd number)", 3, 51, 11, step=2)
        C = st.slider("C (constant)", 0, 10, 2)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
        )

    else:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.image(thresh, caption=f"Result of {method}", use_container_width=True)
