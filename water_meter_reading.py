import streamlit as st
import numpy as np
import cv2
import os
import cv2
from ultralytics import YOLO
import numpy as np
import argparse

### STREAMLIT CONFIGURATION
st.set_page_config(page_title = "YOLO MODEL",layout = "wide")



### MAIN PROGRAM LOGIC
LABELS = [0,1,2,3,4,5,6,7,8,9]
model = YOLO('models/water_meter_model.pt')
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

DIR_FOL = "test"



def preprocess(image,pyramid_levels=3):
    # Sharpening the image for better predictions
    kernel = np.array([[0, -1, 0], 
                        [-1, 5, -1], 
                        [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def process_image(frame):
    results = model(frame)

    # Extract results and draw bounding boxes
    for result in results:
        # Get predictions for bounding boxes, labels, and confidence scores
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class labels
        
        

        for box, conf, cls in zip(boxes, confidences, classes):
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{LABELS[int(cls)]}" #  {conf:.2f}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) , 2)
            cv2.putText(frame, label, (x1+10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    #include image logic here
    st.title("YOLO MODEL")
    files = st.file_uploader(label = "Upload Images Here",type = ["jpg", "png"],accept_multiple_files = True)

    if files:
        for file in files:
            col1, col2 = st.columns(2)
            #savepath = os.path.join(SAVE_DIR, file)
            img_data = file.read()
            img_np = np.frombuffer(img_data, np.uint8)

            # Decode the image using OpenCV
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            with col1:
                st.write("Original Image")
                st.image(img)
            
            img = preprocess(img)

            img = process_image(img)

            with col2:
                
                st.write("Output Image")
                st.image(img)

            

                
