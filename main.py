import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('expression_model.h5')

classes = ['Looking Away', 'bored', 'confused',
           'drowsy', 'engaged', 'frustrated']

# Function to predict expression from an image


def predict_expression(img):
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    return np.argmax(prediction)


# Accessing the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, 1 for external webcam

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Predict expression
        prediction = predict_expression(face_img)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display expression label below the rectangle
        expression = "Expression: " + str(classes[prediction])
        cv2.putText(frame, expression, (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Expression Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
