import shap
import lime
from lime import lime_image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

# Load data function to load and preprocess data
def load_data():
    # Ensure correct paths to your dataset directories
    train_dir = 'd:/Python Projects/Image Classification Project/datasets/train'
    test_dir = 'd:/Python Projects/Image Classification Project/datasets/test'
    
    # Verify that directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Dataset directories not found. Expected paths: {train_dir}, {test_dir}")

    # Data Augmentation settings for better generalization
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Loading training data
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    
    # Loading validation data
    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    # Loading test data
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    
    # Return the data
        # Fetch the first batch from the iterators
    x_train, y_train = next(train_data)  # Use 'next()' or '__next__()' to fetch the batch
    x_test, y_test = next(test_data)

    return x_train, y_train, x_test, y_test


# Train model function to train and save model
def train_model():
    # Assuming you have already trained and saved the model (best.h5)
    model = load_model('../models/best.keras')  # Load the saved model
    
    # Additional training-related code...
    return model


# SHAP Example

def explain_with_shap(model, x_train, x_test):
    # Ensure x_train and x_test are in their original shape: (batch_size, height, width, channels)
    x_train_sample = x_train[:100]  # Use the first 100 samples from x_train
    x_test_sample = x_test[:1]  # Use the first image from x_test
    
    # Remove the batch dimension for a single image (for SHAP to work with)
    x_test_sample_single = np.squeeze(x_test_sample, axis=0)  # Shape becomes (224, 224, 3)
    
    # Flatten the image (from shape (224, 224, 3) to (224*224*3,))
    x_test_sample_single_flattened = x_test_sample_single.flatten()  # Shape becomes (224*224*3,)
    
    # Now use SHAP with the original 4D data for x_train and the flattened x_test image
    explainer = shap.KernelExplainer(model.predict, x_train_sample)
    shap_values = explainer.shap_values(x_test_sample_single_flattened)  # Pass the flattened image here
    
    # Visualize the SHAP values
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0], x_test_sample_single_flattened)


# LIME Example
def explain_with_lime(model, x_test):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x_test[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    explanation.show_in_notebook()


# GradCAM Example
def gradcam(model, x_test):
    grad_model = tf.keras.models.Model([model.inputs], [model.output, model.get_layer("conv2d_3").output])

    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(x_test[0:1], dtype=tf.float32)
        tape.watch(inputs)
        preds, conv_output = grad_model(inputs)
        class_idx = np.argmax(preds[0])
        class_output = preds[:, class_idx]
    
    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = np.mean(conv_output[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + x_test[0]
    cv2.imwrite("gradcam_output.jpg", superimposed_img)


# Main function to run everything
def main():
    # Train and evaluate the model first
    model = train_model()

    # Load sample data (change this based on your dataset)
    x_train, y_train, x_test, y_test = load_data()

    # Call SHAP
    explain_with_shap(model, x_train, x_test)

    # Call LIME
    explain_with_lime(model, x_test)

    # Call GradCAM
    gradcam(model, x_test)


if __name__ == "__main__":
    main()
