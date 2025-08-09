# processing.py
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import zipfile


# pip freeze > requirements.txt
# pip install -r requirements.txt

# Constants
IMAGE_NUMBER = 4000  # Number of images to process
IMAGE_SIZE = 224
ZIP_FILE = "rsna-pneumonia-detection-challenge.zip"  # Path to the zip file
EXTRACT_DIR = "data"  # Directory to extract the zip file contents
DATA_DIR = f"processed_data/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"  # Directory to save processed data

# Ensure output directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Extract the zip file
def extract_zip(zip_file, extract_to):
    """
    Extract a zip file to a specified directory.
    
    Parameters:
        zip_file (str): Path to the zip file.
        extract_to (str): Path to the extraction directory.
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_file} to {extract_to}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_to}.")

# Function to load and preprocess images
def load_images(dicom_dir, labels, num_samples, target_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False):
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    print(f"Total DICOM files found: {len(dicom_files)}")

    datagen = ImageDataGenerator(
        rotation_range=30,  # Increased rotation range
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Add vertical flip for more variation
        fill_mode="nearest",
        brightness_range=[0.6, 1.4],  # Broaden the brightness range
        contrast_range=[0.8, 1.2]  # Add contrast adjustment

    ) if augment else ImageDataGenerator()

    for file_name in tqdm(dicom_files[:num_samples], desc="Processing DICOM Files"):
        file_path = os.path.join(dicom_dir, file_name)
        try:
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array
            img = img / np.max(img)

            img_resized = np.array(Image.fromarray(img).resize(target_size))
            img_resized = np.stack([img_resized] * 3, axis=-1)

            patient_id = os.path.splitext(file_name)[0]
            label = labels.loc[labels['patientId'] == patient_id, 'Target'].values[0]

            img_resized = datagen.random_transform(img_resized)
            yield img_resized, label

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def prepare_data():
    # Step 1: Extract the ZIP file
    extract_zip(ZIP_FILE, EXTRACT_DIR)

    # Step 2: Define paths
    dicom_dir = os.path.join(EXTRACT_DIR, "stage_2_train_images")
    labels_file = os.path.join(EXTRACT_DIR, "stage_2_train_labels.csv")

    # Step 3: Verify the extracted files
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Step 5: Load labels
    labels = pd.read_csv(labels_file)
    print(f"Total labels: {len(labels)}")
    print(f"Label distribution:\n{labels['Target'].value_counts()}")

    # Step 6: Load and preprocess images
    images, labels_filtered = zip(*load_images(dicom_dir, labels, IMAGE_NUMBER))
    images = np.array(images, dtype=np.float32)
    labels_filtered = np.array(labels_filtered, dtype=np.int32)

    print(f"Total images processed: {len(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels_filtered.shape}")

    # Step 7: Split and handle class imbalance
    X_train, X_val, y_train, y_val = train_test_split(images, labels_filtered, test_size=0.2, stratify=labels_filtered)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Step 8: Save validation sets
    print("Saving Validation sets...")
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)

    # Step 9: Apply SMOTE for the training set
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    X_train_res = X_train_res.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    print(f"Resampled training set class distribution:\n{pd.Series(y_train_res).value_counts()}")

    # Step 10: Save training sets
    print("Saving Training sets...")
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_res)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train_res)
    print("Saved Training sets successfully!")

    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
