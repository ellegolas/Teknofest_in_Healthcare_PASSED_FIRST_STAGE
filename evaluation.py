import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, test_images, test_metadata, test_labels):
    """Evaluate model and compute metrics."""
    preds = model.predict([test_images, test_metadata]).round()
    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    cm = confusion_matrix(test_labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm}
