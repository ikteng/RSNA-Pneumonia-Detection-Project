# preprocessing.py
import os
import zipfile
import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow.keras.backend as K
import gc

IMAGE_SIZE = 256
IMAGE_NUMBER = 8000

ZIP_FILE = "rsna-pneumonia-detection-challenge.zip"
EXTRACT_DIR = "data"
MODEL = "efficientnet"
SAVE_DIR = f"processed_data/{MODEL}/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"

os.makedirs(SAVE_DIR, exist_ok=True)

def clear_memory():
    K.clear_session()
    gc.collect()
    print("🧹 Cleared memory!")

clear_memory()

def extract_zip(zip_file, extract_to):
    if not os.path.exists(extract_to):
        print(f"📦 Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction complete.")
    else:
        print(f"✅ Dataset already exists at {extract_to}.")

def load_images(dicom_dir, labels_df, target_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False):
    patient_ids = list(labels_df['patientId'])
    dicom_files = [f"{pid}.dcm" for pid in patient_ids]
    print(f"Processing {len(dicom_files)} DICOM files...")

    def process_file(file_name):
        try:
            file_path = os.path.join(dicom_dir, file_name)
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array.astype(np.float32)
            img = cv2.resize(img, target_size)
            img = np.stack([img] * 3, axis=-1)
            img = preprocess_input(img)

            patient_id = os.path.splitext(file_name)[0]
            label = labels_df.loc[labels_df['patientId'] == patient_id, 'Target'].values[0]

            return img.astype(np.float32), int(label)
        except Exception as e:
            print(f"❌ Error with {file_name}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, dicom_files), total=len(dicom_files)))

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No images were successfully processed.")

    images, labels = zip(*results)
    return list(images), list(labels)

def prepare_data():
    extract_zip(ZIP_FILE, EXTRACT_DIR)

    dicom_dir = os.path.join(EXTRACT_DIR, "stage_2_train_images")
    labels_file = os.path.join(EXTRACT_DIR, "stage_2_train_labels.csv")

    if not os.path.exists(dicom_dir) or not os.path.exists(labels_file):
        raise FileNotFoundError("Required files not found.")

    labels = pd.read_csv(labels_file)
    labels = labels.groupby("patientId")["Target"].max().reset_index()

    available_ids = {f.replace(".dcm", "") for f in os.listdir(dicom_dir) if f.endswith(".dcm")}
    labels = labels[labels['patientId'].isin(available_ids)]

    pos = labels[labels['Target'] == 1]
    neg = labels[labels['Target'] == 0]
    min_class_size = min(len(pos), len(neg), IMAGE_NUMBER // 2)

    pos_sample = pos.sample(min_class_size, random_state=42)
    neg_sample = neg.sample(min_class_size, random_state=42)
    balanced_labels = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Balanced label distribution:\n", balanced_labels['Target'].value_counts())

    images, labels_filtered = load_images(dicom_dir, balanced_labels)

    images = np.array(images, dtype=np.float32)
    labels_filtered = np.array(labels_filtered, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(images, labels_filtered, test_size=0.2, stratify=labels_filtered, random_state=42)
    print("Train label counts:", np.bincount(y_train))
    print("Val label counts:", np.bincount(y_val))

    np.savez_compressed(os.path.join(SAVE_DIR, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(SAVE_DIR, "val_data.npz"), X=X_val, y=y_val)

    print(f"✅ Saved datasets! Train: {len(X_train)}, Val: {len(X_val)}")
    print("🎉 Done.")

prepare_data()
clear_memory()
