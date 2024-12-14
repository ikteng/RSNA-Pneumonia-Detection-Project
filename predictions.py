# predictions.py
import numpy as np
from tensorflow.keras.models import load_model

# Constants
IMAGE_NUMBER = 2000
IMAGE_SIZE = 224
model_type="model1"
version = "version 1"

DATA_DIR = f"{version}/processed_data_{IMAGE_SIZE}-{IMAGE_NUMBER}"
MODEL_DIR1 = f"{version}/model1-{IMAGE_NUMBER}-{IMAGE_SIZE}.keras"
MODEL_DIR2 = f"{version}/model2-{IMAGE_NUMBER}-{IMAGE_SIZE}.keras"
MODEL_DIR=f"{model_type}-{IMAGE_NUMBER}-{IMAGE_SIZE}.keras"

# Load Test Data
print("Loading Test Images...")
X_test = np.load(f"test_images_{IMAGE_SIZE}.npy")  # Adjust filename as needed
print(f"Test Images Loaded: {X_test.shape}")

# Load Pre-trained Models
print("Loading Models...")
model = load_model(MODEL_DIR)
model1 = load_model(MODEL_DIR1)  
model2 = load_model(MODEL_DIR2)
print("Models Loaded Successfully!")

# Summary of the model
print("Summary of Models")
model1.summary()
model2.summary()

def run_prediction():
    # Generate predictions
    print("Generating Predictions on Test Data...")
    test_predictions = model.predict(X_test)
    pred_class = (test_predictions > 0.5).astype(int)

    print("Predictions Generated!")

    # Example output
    print("Sample Predictions:")
    for i in range(5):  # Show a few predictions
        print(f"Predicted: {pred_class[i][0]}, Probability: {test_predictions[i][0]:.4f}")

# run_prediction()

# Function for Ensemble Predictions (Averaging)
def ensemble_predictions(models, X_test):
    """
    Generates ensemble predictions by averaging the outputs of all models.
    
    Parameters:
        models (list): List of trained models.
        X_test (ndarray): Test data.
    
    Returns:
        ndarray: Averaged predictions.
    """
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)

    # Convert to numpy array and compute the mean
    predictions = np.array(predictions)
    ensemble_pred = np.mean(predictions, axis=0)  # Averaging predictions

    return ensemble_pred

# Function for Weighted Ensemble Predictions
def ensemble_predictions_weighted(models, X_test, weights):
    """
    Generates weighted ensemble predictions.
    
    Parameters:
        models (list): List of trained models.
        X_test (ndarray): Test data.
        weights (list): Weights for each model.
    
    Returns:
        ndarray: Weighted averaged predictions.
    """
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)

    # Weighted averaging
    predictions = np.array(predictions)
    weights = np.array(weights).reshape(-1, 1, 1)  # Reshape to match predictions
    ensemble_pred = np.sum(predictions * weights, axis=0) / np.sum(weights)

    return ensemble_pred

def run_ensemble_preidctions():
    # Models List
    models = [model1, model2]

    # Averaging Ensemble Predictions
    print("Generating Averaged Ensemble Predictions...")
    ensemble_pred_avg = ensemble_predictions(models, X_test)

    # Weighted Ensemble Predictions
    weights = [0.4, 0.35, 0.25]  # Example weights based on model performance
    print("Generating Weighted Ensemble Predictions...")
    ensemble_pred_weighted = ensemble_predictions_weighted(models, X_test, weights)

    # Display Predictions
    print("Sample Predictions from Averaged Ensemble:")
    for i in range(5):  # Display the first 5 predictions
        print(f"Image {i+1}: Probability: {ensemble_pred_avg[i][0]:.4f}")

    print("\nSample Predictions from Weighted Ensemble:")
    for i in range(5):  # Display the first 5 predictions
        print(f"Image {i+1}: Probability: {ensemble_pred_weighted[i][0]:.4f}")

run_ensemble_preidctions()