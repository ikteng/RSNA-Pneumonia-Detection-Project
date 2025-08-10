# RSNA Pneumonia Detection Project
Dataset: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

## Overview
This project is focused on developing a pneumonia detection system using deep learning techniques. It aims to analyze chest X-ray images to identify the presence of pneumonia. The project includes several key components for data processing, model building (using DenseNet), model evaluation and prediction

Note: You need to use at most Python 3.11 for the tensorflow library!

## Project Structure and Files
### 1. processing.py
* Data Extraction: The extract_zip function is used to extract a zip file (specified by ZIP_FILE) to a designated directory (EXTRACT_DIR). It checks if the extraction has already been done and proceeds accordingly.
* Image Loading and Preprocessing:
  * load_images function loads DICOM images from a given directory (dicom_dir). It allows for data augmentation (such as rotation, shifting, flipping, adjusting brightness and contrast) based on a specified flag. The function also associates each image with its corresponding label from a provided labels dataframe.
  * load_test_images is dedicated to loading and preprocessing test images. It converts grayscale images to RGB format, normalizes pixel values, resizes them, and saves the processed test images as a .npy file.

* Data Preparation Pipeline: The prepare_data function orchestrates the entire data preparation process. It extracts the data, loads and preprocesses the images, performs EDA, splits the dataset into training and validation sets, handles class imbalance using SMOTE, and finally saves the processed datasets as .npy files in the SAVE_DIR.

### 2. densenet_model.py
* Data Loading: Both files start by loading the training (X_train, y_train) and validation (X_val, y_val) datasets from the processed data directory (DATA_DIR). The shapes of these datasets are then displayed.
* Model Building:
  * In each file, a function (build_model) constructs a neural network model. For densenet_model.py, it's based on the DenseNet121 architecture. The base models are loaded with pre-trained weights from ImageNet, and the last few layers are unfrozen for fine-tuning. Additional layers like global average pooling, batch normalization, dropout, and dense layers are added to form the final model. The model is compiled with an appropriate optimizer (Adam with a low learning rate), loss function (binary cross-entropy), and evaluation metrics (accuracy, precision, recall).
  * Training: A set of callbacks such as early stopping, learning rate reduction, model checkpointing, and a learning rate scheduler (using cosine annealing) are defined. The model is then trained on the training dataset with the defined callbacks and considering class weights to handle any class imbalance. After training, relevant objects and variables are deleted, and garbage collection is forced to free up memory.

### 3. evaluate_model.py
* Model Loading: This script disables GPU usage if required by setting an environment variable, then loads a pre-trained model from the specified path (MODEL_PATH). The model is loaded using TensorFlow/Keras and is ready for evaluation.
* Threshold Tuning & Evaluation: The run_threshold_tuning function evaluates the model on the validation dataset by:
  * Loading preprocessed .npy files for features and labels.
  * Generating predicted probabilities for each sample.
  * Iterating through thresholds from 0.0 to 1.0 (step 0.01) to calculate precision, recall, and F1-score.
  * Selecting the best threshold based on the highest F1-score.
  * Computing the final evaluation metrics using the best threshold:
    * AUC Score
    * Classification Report (Precision, Recall, F1)
    * Confusion Matrix
  * Returning all threshold results for further analysis if needed.

## How to Use
### 1. Data Preparation
Execute the prepare_data function in processing.py. This will handle all the steps related to extracting the dataset, preprocessing the images, splitting the data, and saving the processed datasets for further use. Ensure that the paths for the zip file, extraction directory, and other relevant directories are correctly configured in the code.

### 2. Model Training
For training the DenseNet-based model, run densenet_model.py. The training process will take place with the configured parameters like the number of epochs, batch size, and other hyperparameters. The trained models will be saved to the specified model paths within the code.

### 3. Model Evaluation
Run evaluate_model.py to load a trained model (make sure the MODEL_DIR points to the correct model file) and evaluate it on the validation dataset. The evaluation results will provide insights into how well the model performs in terms of different metrics.


## Dependencies
The project relies on several Python libraries such as os, numpy, pandas, pydicom, PIL (Python Imaging Library), sklearn (for model selection and class weight computation), tqdm (for progress bars), tensorflow (including keras for building and training neural networks), imblearn (for handling class imbalance with SMOTE), matplotlib, and seaborn (for data visualization). Make sure these libraries are installed in your Python environment with appropriate versions to ensure the correct functioning of the project.

Please note that you may need to adjust certain parameters like file paths, number of images to process, model hyperparameters, etc., based on your specific requirements and available computational resources.
