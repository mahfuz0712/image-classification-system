# augmentation starts here
from pytbangla import computer as mahfuz
from src.data_preprocessing import load_data, augment_images
from src.train import train_model
from src.evaluate import evaluate_model



def main():
    x_train, y_train = load_data()
    augmented_images = augment_images(x_train, y_train)
    mahfuz.lekho("Augmented images are ready!")
    mahfuz.lekho(augmented_images)
    # Train the model
    train_model()
    # Evaluate the model after training
    evaluate_model()
   

if __name__ == "__main__":
    main()
