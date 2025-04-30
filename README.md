# CONSTRUCTION_MONITORING

<ol>
  <li> <b>PERSONNEL-ID VERIFICATION</b> : Designed to verify ID-CARD of a personnel working on a construction site with the help of computer vision techniques like OCR, image-preprocessing, face verification. At entry the worker will be prompted to show his ID-CARD in front of a camera. The system with the help of computer vision will extract worker-id, worker face from his ID-CARD and also capture a real-time photo of the worker's face. The worker-id will be verified with pre-existing entries in the database. The ID face and real-time face will be compared for face verification. After all the successful checks, the worker will be authorized to enter the construction site. <br>
    <b>OVERVIEW DIAGRAM</b><br>
    <img src="https://github.com/Parth-D3/CONSTRUCTION_MONITORING/blob/main/util_images/PERSONNEL_ID.png">
  
  <b>KEY POINTS</b>
  <ul>
    <li>HaarCascadeClassifier for face extraction</li>
    <li>DeepFace model for face comparision</li>
    <li>EasyOCR for text extraction</li>
    <li>VGG16 backbone used for face feature extraction</li>
    <img src="https://github.com/Parth-D3/CONSTRUCTION_MONITORING/blob/main/util_images/vgg16_architecture.jpg">
  </ul>
  </li>
  <li><b>Fire Detection:</b> Detect fires on site in case of accidents or electric short circuit malfunctions. Used fine-tuned YOLOv8 model for fire detection.<br>
    <img src="https://github.com/Parth-D3/CONSTRUCTION_SITE_MONITORING/blob/main/util_images/detected_fire.jpeg">
  </li>
</ol>

