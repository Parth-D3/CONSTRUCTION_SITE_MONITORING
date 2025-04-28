import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO(r"models\firebest.pt") 


st.title("Fire and Smoke Detection using YOLOv8")
st.write("Upload an image to detect fire and smoke.")


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.merge([enhanced, enhanced, enhanced])
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    return sharpened


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    processed_image = preprocess_image(image)
    results = model(processed_image)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    st.image(processed_image, caption="Processed Image with Detections", use_column_width=True)
