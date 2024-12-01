import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import load_model # type: ignore
from src.data_preprocessing import load_data




def evaluate_model():
    # Unpack the return values from load_data
    x_train, y_train, x_val, y_val = load_data()  # Assuming load_data now returns 4 variables

    # Evaluate the model using validation data
    model = load_model('../models/best.keras')  # Ensure you're loading the correct model
    loss, accuracy = model.evaluate(x_val, y_val)
    
    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
