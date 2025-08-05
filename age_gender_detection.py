import cv2
import numpy as np

# Load pre-trained models
age_net = cv2.dnn.readNetFromCaffe(
    "D:/Age & Gender Detection Project Using ML/models/age_deploy.prototxt",
    "D:/Age & Gender Detection Project Using ML/models/age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "D:/Age & Gender Detection Project Using ML/models/gender_deploy.prototxt",
    "D:/Age & Gender Detection Project Using ML/models/gender_net.caffemodel"
)

# Labels for age and gender prediction
AGE_LIST = ['(0-2)', '(4-6)', '(10-15)', '(15-20)', '(20-25)', '(25-30)', '(30-40)', '(40-65)', '(65-100)']
GENDER_LIST = ['Male', 'Female']

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def detect_age_gender(face):
    # Prepare the face for the models
    blob = cv2.dnn.blobFromImage(face, 1, (227, 227), 
        (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return gender, age

def process_frame(frame):
    faces = detect_face(frame)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        gender, age = detect_age_gender(face)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (500, 555, 555), 2)
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Access the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    output_frame = process_frame(frame)
    cv2.imshow("Real-time Age and Gender Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
