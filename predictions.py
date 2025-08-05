# predictions.py
import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

IMAGE_SIZE = 224
IMAGE_NUMBER = 8000
EPOCHS = 50
MODEL_NAME = "densenet_model"
MODEL_PATH = f"models/{MODEL_NAME}-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}.keras"
TEST_IMAGE_DIR = os.path.join("data", "stage_2_test_images")
PRED_DIR = "predictions"
os.makedirs(PRED_DIR, exist_ok=True)

# ===== LOAD & PREPROCESS TEST IMAGES =====
def load_test_images():
    dicom_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(".dcm")])
    images = []
    for f in tqdm(dicom_files, desc="Loading DICOM Files"):
        try:
            dicom = pydicom.dcmread(os.path.join(TEST_IMAGE_DIR, f))
            img = cv2.resize(dicom.pixel_array.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE))
            img = np.stack([img] * 3, axis=-1)
            img = preprocess_input(img)
            images.append(img.astype(np.float16))
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")
    return np.array(images, dtype=np.float16), dicom_files

# ===== RUN PREDICTION =====
def run_prediction():
    print("📂 Loading test images...")
    X_test, filenames = load_test_images()
    
    print("🧠 Loading trained model...")
    model = load_model(MODEL_PATH)
    
    print("🔮 Generating predictions...")
    preds = model.predict(X_test, verbose=1)

    # ===== FORMAT 1: Detailed Predictions (binary class + probability) =====
    print("💾 Saving detailed predictions...")
    detailed_data = [
        [f"patient_{i}", int(p[0] > 0.5), float(p[0])]
        for i, p in enumerate(preds)
    ]
    pd.DataFrame(detailed_data, columns=["PatientID", "PredictedClass", "Probability"]) \
        .to_csv(f"{PRED_DIR}/{MODEL_NAME}_detailed.csv", index=False)

    # ===== FORMAT 2: Kaggle Submission (PredictionString with dummy box) =====
    print("💾 Saving Kaggle submission format...")
    submission_data = []
    for fname, p in zip(filenames, preds):
        prob = float(p[0])
        pred_str = f"{prob:.4f} 0 0 100 100" if prob > 0.5 else ""
        submission_data.append([fname.replace(".dcm", ""), pred_str])
    
    pd.DataFrame(submission_data, columns=["patientId", "PredictionString"]) \
        .to_csv(f"{PRED_DIR}/submission.csv", index=False)
    
    print("✅ Predictions completed! Files saved in 'predictions/'")

if __name__ == "__main__":
    run_prediction()
