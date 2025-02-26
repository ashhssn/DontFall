import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = load_model('fall_detection_model.h5')

IMG_SIZE = (224, 224)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict fall probability
    prediction = model.predict(img)
    # For binary classification with sigmoid, a threshold of 0.5 works well
    if prediction[0][0] > 0.5:
        text = "A person has fallen"
    else:
        text = "Normal Pose"

    # Display the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Fall Detection', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
