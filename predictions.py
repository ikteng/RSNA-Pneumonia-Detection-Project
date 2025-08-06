# predictions.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import pandas as pd
from tqdm import tqdm
import pydicom
import cv2

IMAGE_SIZE = 128
IMAGE_NUMBER = 4000
EPOCHS = 30

model = "densenet"
model_name = f"{model}_model"

DATA_DIR = f"processed_data/{model}/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"
MODEL_DIR = f"models/{model_name}-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}.keras"
PRED_DIR = "predictions"
os.makedirs(PRED_DIR, exist_ok=True)

def load_test_images():
    test_image_dir = os.path.join("data", "stage_2_test_images")
    dicom_files = [f for f in os.listdir(test_image_dir) if f.endswith('.dcm')]
    print(f"Total DICOM files found: {len(dicom_files)}")

    test_images = []
    for file_name in tqdm(dicom_files, desc="Loading DICOM Files"):
        file_path = os.path.join(test_image_dir, file_name)
        try:
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array.astype(np.float32)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = np.stack([img] * 3, axis=-1)
            img = preprocess_input(img)
            test_images.append(img.astype(np.float16))
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return np.array(test_images, dtype=np.float16)

# Load the test data
print("Loading the test data...")
X_test = load_test_images()
print("Test data loaded successfully.")

# Load the model
print("Loading the model...")
model = load_model(MODEL_DIR)
print("Model loaded successfully.")

def run_prediction():
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
    prediction_df.to_csv(f"{PRED_DIR}/{model_name}_predictions.csv", index=False)
    print(f"File '{PRED_DIR}/{model_name}_predictions.csv' with detailed predictions has been created successfully!")

if __name__ == "__main__":
    run_prediction()
    print("Predictions Completed!")