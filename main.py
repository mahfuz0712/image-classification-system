# augmentation starts here
from src.data_preprocessing import load_data, augment_images



def main():
    # Load data (images and labels)
    x_train, y_train = load_data()
    
    # Augment images
    augmented_images = augment_images(x_train, y_train)
    
    # Proceed with the model or any other task
    print("Augmented images are ready!")
    
    # You can now use the augmented_images for training your model
    # model.fit(augmented_images, epochs=10)
   



if __name__ == "__main__":
    main()
