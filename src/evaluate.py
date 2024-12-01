import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import load_model # type: ignore
from src.data_preprocessing import load_data


def evaluate_model():
    # Load the trained model
    model = load_model('models/best.h5')  # Ensure the correct path and file format

    # Load test data
    _, test_data = load_data()

    # Extract test images and labels
    x_test, y_test = next(test_data)  # Fetch a batch for evaluation
    x_test = x_test[:len(test_data.labels)]  # Limit to the exact number of samples
    y_test = y_test[:len(test_data.labels)]

    # Predict on test data
    y_pred = model.predict(x_test)

    # Evaluate metrics
    print("Classification Report:")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

    # Compute AUC (example for binary classification)
    if y_test.shape[1] == 2:  # Binary classification
        auc_score = roc_auc_score(y_test[:, 1], y_pred[:, 1])
        print(f"AUC Score: {auc_score}")