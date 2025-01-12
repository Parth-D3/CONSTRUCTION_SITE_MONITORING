import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from deepface import DeepFace
import cv2
import time
import re
import easyocr
import mysql.connector
import numpy as np


path = "data\\captured_image.png"   # to store original ID-CARD
no_noise_path = "data\\no_noise.png" # to store preprocessed ID-CARD
id_face_path = 'data\\extracted_face.jpg'
real_face_path = "data\\realtime_face.png"

# function to capture the initial id card 
def capture_webcam():
    
    cap = cv2.VideoCapture(0) # change '0' to appropriate number if using hd cam

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.imshow("Press Space to Capture Image", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:
            cv2.imwrite(path, frame)
            print("Image captured and saved as 'captured_image.png'")
            break
        
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# function to preprocess id card
def preprocess(path):
    image = cv2.imread(path)
    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
    gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    no_noise_image = cv2.GaussianBlur(binary_image, (5, 5), 0)


    cv2.imwrite(no_noise_path,no_noise_image)

# function to extract worker id
def extract_worker_id(path):
    reader = easyocr.Reader(['en'])  
    result = reader.readtext(path)
    try:
        extracted_text = " ".join([detection[1] for detection in result])
        upper_text = extracted_text.upper()
        match = re.search(r"WORKER ID: (.*?) WORKER NAME: ", upper_text)
        if match:
            id = match.group(1)
            return id
        else:
            print("No match")
            return 0
    except Exception as e:
        print(e)

# function to check if worker id exists in the database
def verify_db(id):
    try:
        conn = mysql.connector.connect(
            host="localhost",      # MySQL server address
            port = 3306,
            user="root",  # MySQL username
            password="password", # ENTER YOU PASSWORD HERE  
            database="WorkerDB"  # The name of your database
        )

        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM worker_info WHERE worker_id = %s"
        cursor.execute(query, (id,))  
        result = cursor.fetchone()
        if conn.is_connected():
            cursor.close()
            conn.close()
        if result[0] > 0:
            return True
        else:
            return False
    except mysql.connector.Error as err:
        print(f"Error: {err}")  

# function to extract face from ID-CARD
def extract_id_face(path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])
        sharpened_image = cv2.filter2D(face, -1, sharpen_kernel)

        norm_img = np.zeros((300, 300))
        norm_img = cv2.normalize(sharpened_image, norm_img, 0, 255, cv2.NORM_MINMAX)

        cv2.imwrite(id_face_path,norm_img)


# function to capture realtim face from webcam
def capture_realtime_face():
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
    
        cv2.imshow("Press Space to Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        
        
        if key == 32:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                sharpen_kernel = np.array([[0, -1, 0],
                                           [-1, 5,-1],
                                           [0, -1, 0]])
                sharpened_image = cv2.filter2D(face, -1, sharpen_kernel)
                norm_img = np.zeros((300, 300))
                norm_img = cv2.normalize(sharpened_image, norm_img, 0, 255, cv2.NORM_MINMAX)

                cv2.imwrite(real_face_path,norm_img)
            break
        
        if key == ord('q'):
            print("Exiting...")
            break

    
    cap.release()
    cv2.destroyAllWindows()

# function to check face similarity
def compare(path1,path2):
    result = DeepFace.verify(path1, path2,threshold=0.5)
    if(result['verified']): print("FACE MATCH SUCCESSFUL !!!")
    else: print("FACE MATCH UNSUCCESSFUL !!!")

print("WELCOME TO PERSONNAL ID VERIFICATION MODULE")

print("HOLD YOUR ID-CARD IN FRONT OF THE CAMERA AND USE SPACE-BAR TO CAPTURE IMAGE")

capture_webcam()
preprocess(path)
worker_id = extract_worker_id(no_noise_path)
id_verified = verify_db(worker_id.strip())
if id_verified:
    extract_id_face(no_noise_path)
    capture_realtime_face()
    compare(id_face_path,real_face_path)
else:
    print("Worker ID Does Not exist in db")
    







