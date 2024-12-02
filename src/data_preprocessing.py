# src/data_preprocessing.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np



def load_data():
    # Initialize data generators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load training data
    train_data = datagen.flow_from_directory(
        'datasets/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    # Load testing data
    test_data = datagen.flow_from_directory(
        'datasets/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    # Convert to numpy arrays
    x_train, y_train = next(train_data)
    x_test, y_test = next(test_data)

    return x_train, y_train, x_test, y_test

# Function to augment images
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
    datagen.fit(x_train)
    
    # Create augmented images
    augmented_images = datagen.flow(x_train, y_train, batch_size=16)
    
    return augmented_images

# If needed, you can call these functions here for testing
if __name__ == '__main__':
    x_train, y_train = load_data()
    augmented_images = augment_images(x_train, y_train)
    print("Augmented images are ready!")
    print(augmented_images)
