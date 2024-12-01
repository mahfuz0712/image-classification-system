import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import load_model # type: ignore
from src.data_preprocessing import load_data


def evaluate_model():
    # Load the trained model
    model = load_model('../models/best.h5')  # Ensure correct path

    # Load test data
    _, test_data = load_data()

    # Evaluate metrics
    y_pred = model.predict(test_data)

    # Evaluate and print classification report
    print("Classification Report:")
    print(classification_report(test_data.labels, y_pred))
