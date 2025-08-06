# build_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
import gc  # garbage collector
import tensorflow.keras.backend as K

# Disable ONEDNN for CPU compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMAGE_SIZE = 224
IMAGE_NUMBER = 4000
EPOCHS = 30
BATCH_SIZE = 64

model = "densenet"
model_name = f"{model}_model"

DATA_DIR = f"processed_data/{model}/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"
MODEL_PATH = f"models/{model_name}-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}.keras"

def clear_memory():
    K.clear_session()
    gc.collect()
    print("🧹 Cleared memory!")

clear_memory()

# Load data
print("Loading data...")
train_data = np.load(os.path.join(DATA_DIR, "train_data.npz"))
X_train = train_data["X"]
y_train = train_data["y"]

val_data = np.load(os.path.join(DATA_DIR, "val_data.npz"))
X_val = val_data["X"]
y_val = val_data["y"]

# Display data shapes
# print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"Validation set: X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

print("Building model...")
# Build the model
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), l2_reg=0.001):
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    # base_model.trainable = False  # Freeze all layers

    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-100:]:
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-4)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train model
print("Training model...")

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)
print("Completed Training!")

print("Evaluating Model...")
evaluation_metrics = model.evaluate(val_dataset, verbose=1)
print(f"Evaluation Results:\nLoss = {evaluation_metrics[0]:.4f}, Accuracy = {evaluation_metrics[1]:.4f}")

clear_memory()