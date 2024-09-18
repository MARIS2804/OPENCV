import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit file uploader
img_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Check if an image is uploaded
if img_file is not None:
    # Convert the file to an OpenCV image
    img_pil = Image.open(img_file)
    img_cv = np.array(img_pil)

    # Select an operation to perform
    reply = st.selectbox("Select an operation", ["TO RGB", "TO GRAY", "RESIZE", "SHARPENING", "BLURRING", "ERODE", "DILATION", "EDGE DETECTION"])

    # TO RGB
    if reply == "TO RGB":
        fig, ax = plt.subplots(1, 3, figsize=(7, 7))
        ax[0].imshow(img_cv[:, :, 0], cmap="Reds")
        ax[1].imshow(img_cv[:, :, 1], cmap="Greens")
        ax[2].imshow(img_cv[:, :, 2], cmap="Blues")
        for i in range(3):
            ax[i].axis('off')
        st.pyplot(fig)

    # TO GRAY
    elif reply == "TO GRAY":
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_gray, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    # RESIZE
    elif reply == "RESIZE":
        new_size = st.slider("Select new size (pixels)", 10, 1000, 100)
        img_resize = cv2.resize(img_cv, (new_size, new_size))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_resize)
        ax.axis("off")
        st.pyplot(fig)

    # SHARPENING
    elif reply == "SHARPENING":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_sharp = cv2.filter2D(img_cv, -1, kernel)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_sharp)
        ax.axis("off")
        st.pyplot(fig)

    # EDGE DETECTION
    elif reply == "EDGE DETECTION":
        edges = cv2.Canny(img_cv, 100, 200)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(edges, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    # BLURRING
    elif reply == "BLURRING":
        kernel = np.ones((5, 5), np.float32) / 25
        img_blur = cv2.filter2D(img_cv, -1, kernel)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_blur)
        ax.axis("off")
        st.pyplot(fig)

    # ERODE
    elif reply == "ERODE":
        kernel = np.ones((3, 3), np.uint8)
        img_erode = cv2.erode(img_cv, kernel, iterations=1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_erode)
        ax.axis("off")
        st.pyplot(fig)

    # DILATION
    elif reply == "DILATION":
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_cv, kernel, iterations=1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_dilate)
        ax.axis("off")
        st.pyplot(fig)

else:
    st.write("Please upload an image.")
