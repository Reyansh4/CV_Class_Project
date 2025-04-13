import streamlit as st
import numpy as np
import cv2
from undistort_calibration import calibrate_from_images_using_undistortion
from remap_calibration import calibrate_from_images_using_remapping

st.set_page_config(page_title="Camera Calibration", layout="wide")
st.title("📸 Camera Calibration App")

uploaded_files = st.file_uploader("Upload checkerboard images (.tif)", type=["tif"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"{len(uploaded_files)} images uploaded. Starting calibration...")

    # Read and decode uploaded images
    images = []
    for file in uploaded_files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images.append(img)

    st.subheader("🛠 Select the Method Method")
    method = st.selectbox(
            "Choose the undistortion method",
            ("Method A: Undistortion", "Method B: Remapping")
    )

    if method == "Method A: Undistortion":
        result = calibrate_from_images_using_undistortion(images)

        if result:
            st.subheader("📐 Calibration Results")
            st.code(f"Camera Matrix:\n{result['camera_matrix']}")
            st.code(f"Distortion Coefficients:\n{result['dist_coeffs'].flatten()}")
            st.code(f"Reprojection Error: {result['error']:.4f}")

            st.subheader("🖼 Corner Detection Preview")
            cols = st.columns(3)
            for i, img in enumerate(result["images"]):
                with cols[i % 3]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Image {i+1}")

            st.subheader("🛠 Undistortion Method")
            st.markdown("### 🧪 Comparison: Original vs Undistorted")

            st.image([
                cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(result["undistorted"], cv2.COLOR_BGR2RGB)
            ], caption=["Original", "Undistorted (Full Frame)"], width=400)

            st.markdown("### ✂️ Cropped Undistorted Image")
            st.image(cv2.cvtColor(result["undistorted_cropped"], cv2.COLOR_BGR2RGB), caption="Cropped Undistorted", width=400)
        else:
            st.error("Failed to calibrate the camera in the undistorted method. Please check the images and try again.")

    else:
        result = calibrate_from_images_using_remapping(images)

        if result:
            st.subheader("📐 Calibration Results")
            st.code(f"Camera Matrix:\n{result['camera_matrix']}")
            st.code(f"Distortion Coefficients:\n{result['dist_coeffs'].flatten()}")
            st.code(f"Reprojection Error: {result['error']:.4f}")

            st.subheader("🖼 Corner Detection Preview")
            cols = st.columns(3)
            for i, img in enumerate(result["images"]):
                with cols[i % 3]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Image {i+1}")

            st.subheader("🛠 Remapping Method")
            st.markdown("### 🧪 Comparison: Original vs Remapped")

            st.image([
                cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(result["remapped"], cv2.COLOR_BGR2RGB)
            ], caption=["Original", "Remapped (Full Frame)"], width=400)

            st.markdown("### ✂️ Cropped Remapped Image")
            st.image(cv2.cvtColor(result["remapped_cropped"], cv2.COLOR_BGR2RGB), caption="Cropped Remapped", width=400)
        else:
            st.error("Failed to calibrate the camera in the remapped method. Please check the images and try again.")
