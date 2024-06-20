import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Specify the path to the model file
model_path = 'model.keras'  

# Print the model path for debugging
print("Model Path:", model_path)

# Load the trained model
try:
    model = load_model(model_path)
except OSError as e:
    print("Error: Unable to load the model. Please check the file path and ensure the model file exists.")
    exit()

# Constants
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
alphabet = ['A', 'B', 'C', 'D']  

# Function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess image
    processed_img = preprocess_image(frame)

    # Predict
    prediction = model.predict(processed_img)
    predicted_label = alphabet[np.argmax(prediction)]
    confidence_score = np.max(prediction)

    # Print prediction and confidence score
    print("Predicted Label:", predicted_label)
    print("Confidence Score:", confidence_score)

    # Display prediction
    cv2.putText(frame, f"{predicted_label} ({confidence_score:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
