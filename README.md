# RSNA Pneumonia Detection Project
Dataset: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

## Overview
This project is focused on developing a pneumonia detection system using deep learning techniques. It aims to analyze chest X-ray images to identify the presence of pneumonia. The project includes several key components for data processing, model building (using DenseNet), model evaluation, and prediction.

Note: You need to use at most Python 3.11 for the tensorflow library!

## Project Structure and Files
### 1. processing.py
This handles data extraction, preprocessing, and saving of the RSNA Pneumonia Detection Challenge dataset, converting DICOM images into a clean and balanced dataset ready for training deep learning models.

- Automatic ZIP extraction of the RSNA dataset if not already extracted
- Sample equal number of positive (pneumonia) and negative cases to create a smaller, balanced dataset
- DICOM -> Image processing: loads .dcm files using pydicom, converts them from grayscale to RGB, resizes them to a specified image size, and applies preprocess_input() on those images
- Multithreaded image loading using ThreadPoolExecutor for faster processing
- Uses gc and Tensorflow's clear_session to manage memory
- Train/Validation split using train_test_split() (80/20), stratified by label
- Saves datasets as compressed .npz files for fast I/O in training

### 2. build_model.py
This script builds and trains a convolutional neural network for pneumonia detection using transfer learning with deep learning models. It loads preprocessed image data, trains the model using class balancing and augmentation techniques, and saves the best model based on validation accuracy.

- Loads training and validation data generated by preprocessing.py
- Constructs a fine-tuned DenseNet121 model with additional dense layers
- Applies image augmentation (flip, brightness, contrast)
- Computes class weights to handle class imbalance
- Uses callbacks for early stopping, learning rate reduction, and checkpoint saving
- Trains the model and evaluates final validation performance
- Saves the best-performing model to models/

Note: You can get different results when you run this program multiple times!

### 3. evaluate_model.py
This script evaluates a trained deep learning model (e.g., DenseNet) on the validation dataset created during preprocessing. It provides detailed performance metrics to help understand model effectiveness on detecting pneumonia.

- Loads a trained model saved by build_model.py
- Loads the validation set generated by preprocessing.py
- Runs predictions and evaluates the model using: 
  - AUC Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

### 4. predictions.py
This script uses a trained model to make predictions on the RSNA Pneumonia Detection Challenge test set. It processes raw DICOM images, applies the trained model, and saves the predictions with both predicted class labels and confidence scores.

- Loads DICOM test images from the stage_2_test_images directory
- Preprocesses images using preprocess_input for DenseNet
- Loads the trained model
- Generates binary predictions (0 or 1) and associated probabilities
- Saves predictions to a CSV file

# How to Use This Repository
1. Clone the repository
  ```bash
  git clone https://github.com/ikteng/RSNA-Pneumonia-Detection-Project.git
  cd RSNA-Pneumonia-Detection-Project
  ```

2. Create a virtual environment and install dependencies
  ```bash
  pip install -r requirements.txt
  pip install --no-cache-dir tensorflow imblearn
  ```

3. Modify parameters as needed and have fun running the project!

Please note that you may need to adjust certain parameters like file paths, number of images to process, model hyperparameters, etc., based on your specific requirements and available computational resources.
