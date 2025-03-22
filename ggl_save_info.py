import json
import numpy as np
import os

# Path to save model metrics in Google Drive
METRICS_FOLDER = "/content/drive/MyDrive/metrics/"
os.makedirs(METRICS_FOLDER, exist_ok=True)  # Ensure the folder exists

def save_model_info(model, metrics, epochs, learning_rates):
    """Save simplified model information including metrics, learning rate, and epochs to Google Drive."""
    info_file = os.path.join(METRICS_FOLDER, "model_info.json")
    
    # Convert NumPy arrays inside metrics to lists and remove None values
    metrics_serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
        if value is not None
    }
    
    # Simplified model summary
    model_summary = {
        "layers": len(model.layers),
        "trainable_params": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
        "non_trainable_params": int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    }
    
    # Collecting all data
    model_info = {
        "epochs": epochs,
        "initial_learning_rate": learning_rates[0],
        "final_learning_rate": learning_rates[1],
        "metrics": metrics_serializable,
        "model_summary": model_summary
    }
    
    # Append or create new file
    if os.path.exists(info_file):
        with open(info_file, "r+") as json_file:
            existing_data = json.load(json_file)
            existing_data.append(model_info)
            json_file.seek(0)
            json.dump(existing_data, json_file, indent=4)
    else:
        with open(info_file, "w") as json_file:
            json.dump([model_info], json_file, indent=4)
    
    print(f"Model info saved to Google Drive: {info_file}")

    # logs are saved in /content/drive/MyDrive/metrics/model_info.json

