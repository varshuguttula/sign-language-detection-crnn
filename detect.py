import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Define directories
base_dir = 'C:/Users/alaha/OneDrive/Documents/itwillrun'
train_dir = os.path.join(base_dir, 'train_data')
val_dir = os.path.join(base_dir, 'validation_data')

# Get the class names from the folder names
CLASSES = sorted(os.listdir(train_dir))

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define a frame skip counter
frame_skip = 0
frame_skip_threshold = 5  # Adjust this value to skip frames

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_skip += 1
    if frame_skip < frame_skip_threshold:
        continue
    frame_skip = 0
    
    # Preprocess the frame
    frame_resized = cv2.resize(frame, (128, 128))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)
    
    # Predict the gesture
    predictions = model.predict(frame_reshaped)
    predicted_class = np.argmax(predictions)
    predicted_label = CLASSES[predicted_class]
    
    # Display the prediction
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
