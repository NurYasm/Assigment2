import cv2
from deepface import DeepFace
import os

# Disable OpenCL
cv2.ocl.setUseOpenCL(False)
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Path to the image file
path = r'C:\Users\HP\Desktop\Assignment\picture.jpg'
# Read the image
image = cv2.imread(path)
# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Convert grayscale image to RGB format for DeepFace
rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Process each detected face
for (x, y, w, h) in faces:
    # Extract the face ROI (Region of Interest)
    face_roi = rgb_image[y:y + h, x:x + w]

    # Perform emotion analysis on the face ROI
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

    # Handle multiple faces returned by DeepFace.analyze
    if isinstance(result, list):
        for res in result:
            emotion = res['dominant_emotion']
            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    else:
        emotion = result['dominant_emotion']
        # Draw rectangle around face 
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

# Display the result
cv2.imshow('Emotion displayed', img)
cv2.waitKey(0)==ord("q")
cv2.destroyAllWindows()
