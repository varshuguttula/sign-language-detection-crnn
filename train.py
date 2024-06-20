import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define constants
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 30 
EPOCHS = 5
NUM_CLASSES = 4  # Updated number of classes
LEARNING_RATE = 0.0002  # Define the learning rate

# Define directories
base_dir = 'C:/Users/alaha/OneDrive/Documents/itwillrun'
train_dir = os.path.join(base_dir, 'train_data')
val_dir = os.path.join(base_dir, 'validation_data')

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    classes=['A', 'B', 'C', 'D'])

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical',
                                                classes=['A', 'B', 'C', 'D'])

# Build CRNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten and LSTM layers
model.add(layers.Flatten())
model.add(layers.Reshape((450, 128)))  # Adjusted Reshape layer

model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(64))

# Dense layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile the model with specified learning rate
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=val_generator)

# Save the model
model.save('model.keras')

# Display accuracy and test loss
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Print accuracy and test loss values
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Confusion Matrix
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_generator.classes, y_pred)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(val_generator.classes, y_pred, target_names=['A', 'B', 'C', 'D']))

# Plot Accuracy and Loss Trends
plot_history(history)

# Heatmap for Accuracy of Train and Validation
accuracy_heatmap = np.array([[history.history['accuracy'][-1], history.history['val_accuracy'][-1]]])
plt.figure(figsize=(6, 4))
sns.heatmap(accuracy_heatmap, annot=True, fmt=".4f", cmap="coolwarm", xticklabels=['Training', 'Validation'], yticklabels=False)
plt.title('Accuracy of Train and Validation')
plt.xlabel('Accuracy')
plt.ylabel('')
plt.show()

# Bar Plot for Train and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Training', 'Validation'], [history.history['accuracy'][-1], history.history['val_accuracy'][-1]], color=['blue', 'orange'])
plt.title('Train and Validation Accuracy')
plt.ylabel('Accuracy')
plt.show()

# Scatter Plot for Train and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.scatter(['Training'], history.history['accuracy'][-1], label='Training Accuracy', color='blue')
plt.scatter(['Validation'], history.history['val_accuracy'][-1], label='Validation Accuracy', color='orange')
plt.title('Train and Validation Accuracy')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
