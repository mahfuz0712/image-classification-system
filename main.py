
from src.data_preprocessing import load_data, augment_images
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Load the training data
    x_train, y_train, x_test, y_test = load_data()  # Assuming load_data now returns 4 values

    # Augment the training images
    augmented_images = augment_images(x_train, y_train)
    print("Augmented images are ready!")

    # Print the augmented image data (optional, for debugging or verification)
    print(str(augmented_images))  # Ensuring that it's a string to avoid potential issues

    # Train the model
    train_model()

    # Evaluate the model after training
    evaluate_model()

if __name__ == "__main__":
    main()
