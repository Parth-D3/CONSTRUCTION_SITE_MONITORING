import streamlit as st
import easyocr
import re
import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# Initialize EasyOCR reader (for English only)
reader = easyocr.Reader(['en'])

# Connect to SQLite database (or create it)
conn = sqlite3.connect('worker_database.db')
cursor = conn.cursor()

# Create table (if it doesn't exist)
cursor.execute('''CREATE TABLE IF NOT EXISTS workers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT UNIQUE,
                    name TEXT,
                    gender TEXT,
                    dob TEXT
                )''')
conn.commit()

# Function to preprocess image and extract text using EasyOCR
def process_image(image):
    #st.write("**Original Image**")
    #st.image(image, caption="Original Image", channels="BGR")

    # Resize image to improve OCR accuracy
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #st.write("**Resized Image**")
    #st.image(resized_image, caption="Resized Image", channels="BGR")

    # Convert to grayscale using OpenCV
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    #st.write("**Grayscale Image**")
    #st.image(gray_image, caption="Grayscale Image", channels="GRAY")

    # Apply noise removal with morphological opening and a closing operation
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    #st.write("**Denoised Image**")
    #st.image(denoised_image, caption="Denoised Image", channels="GRAY")

    # Perform OCR using EasyOCR
    results = reader.readtext(denoised_image, detail=0)
    extracted_text = " ".join(results)

    st.write("**Extracted Text:**")
    #st.write(extracted_text)

    # Update regex pattern to better capture the format including an '@' symbol
    worker_id_pattern = re.compile(r'WORKER\s*ID[:\s]*([A-Z0-9@]+)', re.IGNORECASE)

    # Search for the worker ID in the extracted text
    worker_id_match = worker_id_pattern.search(extracted_text)
    if worker_id_match:
        worker_id = worker_id_match.group(1)  # Extract the Worker ID as it appears

        # Normalize the Worker ID
        worker_id = re.sub(r'[^A-Za-z0-9@]', '', worker_id)  # Retain alphanumeric characters and '@'
        
        # Ensure it's in uppercase
        worker_id = worker_id.upper()

        return worker_id
    else:
        return None

# Function to detect and extract face
def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    padding_height = 0.32
    padding_width = -0.02

    face_image = None
    for (x, y, w, h) in faces:
        x_new = max(0, x - int(w * padding_width))
        y_new = max(0, y - int(h * padding_height))
        w_new = min(image.shape[1] - x_new, int(w * (1 + 2 * padding_width)))
        h_new = min(image.shape[0] - y_new, int(h * (1 + 2 * padding_height)))
        face_image = image[y_new:y_new + h_new, x_new:x_new + w_new]
        break  # Extract only the first detected face

    return face_image

# Function to add a new worker to the database
def add_worker(worker_id, name, gender, dob):
    try:
        cursor.execute("INSERT INTO workers (worker_id, name, gender, dob) VALUES (?, ?, ?, ?)",
                       (worker_id, name, gender, dob))
        conn.commit()
        st.write(f"Worker {name} added successfully!")
    except sqlite3.IntegrityError:
        st.write(f"Worker ID {worker_id} already exists.")

# Function to check if the worker exists in the database
def check_worker(worker_id):
    cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
    result = cursor.fetchone()  # Fetch one result
    if result:
        st.write(f"Identity Verified: {result}")
        return True
    else:
        st.write(f"No matching worker found for Worker ID: {worker_id}")
        return False

# Streamlit app
st.title("ID Card Scanner")

# Initialize session state variables
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None

# Display buttons
st.write("**Start Webcam**")
if st.button("Start Webcam"):
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.write("Webcam started.")
    else:
        st.write("Webcam is already running.")

st.write("**Capture Image**")
if st.button("Capture Image"):
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if ret:
            st.session_state.captured_frame = frame
            st.image(st.session_state.captured_frame, caption="Captured Image", channels="BGR")
            st.write("Image Captured!")

            # Save the captured image to the desktop
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "captured_image.jpg")
            cv2.imwrite(desktop_path, frame)
            st.write(f"Captured image saved at {desktop_path}.")
    else:
        st.write("Please start the webcam first.")

st.write("**Analyze Image**")
if st.button("Analyze Image"):
    if st.session_state.captured_frame is not None:
        # Extract worker ID
        worker_id = process_image(st.session_state.captured_frame)
        if worker_id:
            st.write(f"Detected Worker ID: {worker_id}")

            # Check if Worker ID exists in the database
            check_worker(worker_id)
        else:
            st.write("No Worker ID detected.")
        
        # Extract face
        face_image = extract_face(st.session_state.captured_frame)
        if face_image is not None:
            st.write("**Extracted Face**")
            st.image(face_image, caption="Extracted Face", channels="BGR")
            
            # Save the extracted face image
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "extracted_face.jpg")
            cv2.imwrite(desktop_path, face_image)
            st.write(f"Face image saved at {desktop_path}.")
        else:
            st.write("No face detected.")
    else:
        st.write("No image captured yet. Please capture an image first.")

st.write("**Stop Webcam**")
if st.button("Stop Webcam"):
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        st.write("Webcam stopped.")
    else:
        st.write("Webcam is not running.")

# # Adding example workers
# if st.button("Add Example Workers"):
#     add_worker("AIOP@12345", "John Doe", "Male", "01/01/1990")
#     add_worker("AIOP@67890", "Jane Smith", "Male", "05/15/1985")
#     add_worker("AIOP@95412", "William Harris", "Male", "05/15/1993")
#     add_worker("AIOP@62354", "Michael Turner", "Male", "10/02/1991")

# # Text input to check worker identity
# worker_id_to_check = st.text_input("Enter Worker ID to check:", "AIOP@12345")
# if st.button("Check Worker ID"):
#     check_worker(worker_id_to_check)

# # Display webcam feed if running
# if st.session_state.cap is not None and st.session_state.cap.isOpened():
#     ret, frame = st.session_state.cap.read()
#     if ret:
#         st.image(frame, channels="BGR", caption="Webcam Feed")
