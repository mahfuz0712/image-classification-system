from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load your data here (e.g., x_train, y_train, etc.)
# Assuming you have already loaded your images and labels as NumPy arrays
# For example, use ImageDataGenerator for loading and augmenting images in real-time

# Example function to augment images
def augment_images(x_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Fit the data generator to the training data
    if isinstance(x_train, np.ndarray):  # Ensure x_train is a NumPy array
        datagen.fit(x_train)
    else:
        raise ValueError("x_train must be a numpy array")

    augmented_images = datagen.flow(x_train, y_train, batch_size=16)
    
    return augmented_images
