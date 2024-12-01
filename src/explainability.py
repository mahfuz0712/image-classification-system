import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def explain_model():
    # Load the trained model
    model = load_model('best.h5')  # Ensure correct extension

    # Load a sample image (example)
    img_path = 'path_to_your_sample_image.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    # Generate Grad-CAM
    heatmap = grad_cam(model, img_array, 'last_conv_layer_name')
    plt.matshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.show()
