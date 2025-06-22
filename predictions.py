# predictions.py
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Constants for configuring the data and model paths
IMAGE_SIZE = 224
EPOCHS = 30

DATA_DIR = f"processed_data/processed_data_{IMAGE_SIZE}"
DENSENET_MODEL_DIR = f"models/densenet/densenet_model-{IMAGE_SIZE}-{EPOCHS}.keras"

# Load the test data
print("Loading the test data...")
X_test = np.load(f"test_images_{IMAGE_SIZE}.npy")
print("Test data loaded successfully.")

# Load the pre-trained models
print("Loading the pre-trained models...")
# model = load_model(MODEL_DIR)
densenet_model = load_model(DENSENET_MODEL_DIR)
# resnet_model = load_model(RESENET_MODEL_DIR)
print("Pre-trained models loaded successfully.")

def run_prediction():
    model = densenet_model
    model_name = "densnet_model"

    print(f"Generating predictions using {model_name}...")
    test_predictions = model.predict(X_test)
    pred_class = (test_predictions > 0.5).astype(int)
    print(f"Predictions using {model_name} generated successfully.")

    # Generate patient IDs based on index (you can change this if you have other ID generation logic)
    patient_ids = [f"patient_{i}" for i in range(len(X_test))]

    # Prepare data for the new file with detailed predictions
    prediction_data = []
    print(f"Preparing detailed prediction data for {model_name} predictions...")
    for i in range(len(X_test)):
        patient_id = patient_ids[i]
        predicted_class = pred_class[i][0]
        probability = test_predictions[i][0]
        prediction_data.append([patient_id, predicted_class, probability])

    print("Creating a DataFrame with detailed prediction data for single model predictions...")
    columns = ["PatientID", "PredictedClass", "Probability"]
    prediction_df = pd.DataFrame(prediction_data, columns=columns)

    print("Saving detailed prediction data for single model predictions to a file...")
    prediction_df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"File '{model_name}_predictions.csv' with detailed predictions has been created successfully!")

if __name__ == "__main__":
    run_prediction()
    print("Predictions Completed!")