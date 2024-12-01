import shap
import lime
from lime import lime_image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

# Your previously defined function for training the model
def train_model():
    # Assuming you have already trained and saved the model (best.h5)
    model = load_model('models/best.h5')  # Load the saved model
    
    # Additional training-related code...
    return model

# SHAP Example
def explain_with_shap(model, x_train, x_test):
    explainer = shap.KernelExplainer(model.predict, x_train[:100])
    shap_values = explainer.shap_values(x_test[:1])
    
    # Visualize the SHAP values
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0], x_test[0])

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
