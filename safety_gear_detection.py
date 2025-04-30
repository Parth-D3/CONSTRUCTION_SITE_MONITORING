import cv2
from ultralytics import YOLO


model_path = r"models\best.pt"
model = YOLO(model_path)

# ======================================================================
# Helper functions for evaluating if a person is fully equipped
# ======================================================================

def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Area (IoA) of boxB with respect to boxA.
    Here, boxA is assumed to be the person's bounding box and boxB is the equipment.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxBArea == 0:
        return 0.0
    return interArea / boxBArea

def check_equipment_for_person(person_box, equipment_boxes, threshold=0.5):
    """
    Checks if the required equipment are sufficiently inside the person bounding box.
    Only these equipment are checked:
       - 'Safety Helmet'
       - 'Safety waist'
       - 'safety shoes'
    Returns a dictionary with Boolean flags for each.
    """
    required_equipment = {
        'Safety Helmet': False,
        'Safety waist': False,
        'safety shoes': False
    }
    
    for label, eq_box in equipment_boxes:
        if label not in required_equipment:
            continue
        ioa = compute_iou(person_box, eq_box)
        if ioa >= threshold:
            required_equipment[label] = True

    return required_equipment

def process_detections(detections, threshold=0.75):
    """
    Processes all detections and for each detected person checks if
    all required equipment is mostly within the person's bounding box.
    
    detections: list of tuples (label, confidence, [x1, y1, x2, y2])
    
    Returns a list of dictionaries for each person with:
       - person_box: Bounding box of the person
       - equipment_status: Dictionary with equipment check results
       - fully_equipped: True if the person has all equipment, else False.
    """
    # Filter out persons and equipment detections.
    persons = [det for det in detections if det[0] == 'person']
    equipment = [(label, bbox) for label, conf, bbox in detections if label != 'person']
    
    results = []
    for person in persons:
        label, conf, pbox = person
        person_equipment = check_equipment_for_person(pbox, equipment, threshold)
        results.append({
            "person_box": pbox,
            "equipment_status": person_equipment,
            "fully_equipped": all(person_equipment.values())
        })
    
    return results

# ======================================================================
# Detection & Annotation Functions for Image, Video, and Real-Time
# ======================================================================

def annotate_frame(frame):
    """
    Runs detection on the frame, then annotates it by:
      - Drawing each detection (equipment and person).
      - Checking each person’s equipment.
      - Redrawing the person bounding box with a status label.
    """

    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            detections.append((label, confidence, [x1, y1, x2, y2]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 200), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
    
    # Process persons to check for required safety equipment.
    person_results = process_detections(detections)
    
    for res in person_results:
        x1, y1, x2, y2 = res["person_box"]
        if res["fully_equipped"]:
            color = (0, 255, 0)  # Green for fully equipped.
            status_text = "Fully Equipped"
        else:
            color = (0, 0, 255)  # Red for not fully equipped.
            status_text = "Not Fully Equipped"
        # Redraw the person bbox with status color.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status_text, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def detect_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return
    
    annotated_img = annotate_frame(img)
    cv2.imshow("Safety Gear Detection - Image", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video:", video_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = annotate_frame(frame)
        cv2.imshow("Safety Gear Detection - Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_real_time():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame = annotate_frame(frame)
        cv2.imshow("Real-Time Safety Gear Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ======================================================================
# Main execution: select mode and run
# ======================================================================

mode = input("Enter mode (image/video/real-time): ").strip().lower()

if mode == "image":
    image_path = input("Enter image path: ").strip()
    detect_in_image(image_path)
elif mode == "video":
    video_path = input("Enter video path: ").strip()
    detect_in_video(video_path)
elif mode == "real-time":
    detect_real_time()
else:
    print("Invalid mode! Choose 'image', 'video', or 'real-time'.")
