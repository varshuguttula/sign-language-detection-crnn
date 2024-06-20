import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator with increased intensity
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.5, 1.5],  # Random brightness changes
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directory containing original images
original_dir = 'C:/Users/alaha/OneDrive/Documents/itwillrun/train_data/B'

# Directory to save augmented images
augmented_dir = 'C:/Users/alaha/OneDrive/Documents/itwillrun/train_data/B'

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Augment each image in the directoryx``
for filename in os.listdir(original_dir):
    img_path = os.path.join(original_dir, filename)
    img = image.load_img(img_path, target_size=(128, 128), color_mode="grayscale")  # Convert to grayscale
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix='temp', save_format='jpg'):
        i += 1
        if i >= 40:  # Generate 40 augmented images per original image
            break
    
    # Rename augmented images
    temp_files = [f for f in os.listdir(augmented_dir) if f.startswith('temp')]
    for idx, temp_file in enumerate(temp_files):
        new_name = f"{filename.split('.')[0]}_{idx + 1}.jpg"
        os.rename(os.path.join(augmented_dir, temp_file), os.path.join(augmented_dir, new_name))
