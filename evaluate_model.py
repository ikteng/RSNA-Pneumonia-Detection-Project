# evaluate_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from preprocessing import DATA_DIR
from densenet_model import MODEL_PATH

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load model
print("Loading Model...")
print("Model: ", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model Loaded Successfully!")

def run_threshold_tuning():
   print("Loading Validation Data...")
   X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
   y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
   print(f"Validation Data Loaded: {X_val.shape}, {y_val.shape}")

   # Get predicted probabilities
   print("Generating Predictions...")
   val_probs = model.predict(X_val).ravel()

   best_threshold = 0.5
   best_f1 = 0
   results = []

   for threshold in np.arange(0.0, 1.01, 0.01):
      preds = (val_probs >= threshold).astype(int)
      recall = recall_score(y_val, preds, zero_division=0)
      precision = precision_score(y_val, preds, zero_division=0)
      f1 = f1_score(y_val, preds, zero_division=0)
      results.append((threshold, recall, precision, f1))

      if f1 > best_f1:
         best_f1 = f1
         best_threshold = threshold

   # Evaluate with best threshold
   final_preds = (val_probs >= best_threshold).astype(int)
   auc_score = roc_auc_score(y_val, val_probs)
   cm = confusion_matrix(y_val, final_preds)
   report = classification_report(y_val, final_preds, target_names=["No Pneumonia", "Pneumonia"])

   print("\nBest Threshold (Max F1):", round(best_threshold, 2))
   print("Best F1 Score:", round(best_f1, 4))
   print("AUC Score:", round(auc_score, 4))
   print("\nClassification Report:\n", report)
   print("Confusion Matrix:\n", cm)

   return results

if __name__ == "__main__":
   # model.summary()
   tuning_results = run_threshold_tuning()
