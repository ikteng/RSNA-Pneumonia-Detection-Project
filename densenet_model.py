# densenet_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import gc
from tensorflow.keras import backend as K
from preprocessing import IMAGE_NUMBER, IMAGE_SIZE, DATA_DIR


EPOCHS = 30
BATCH_SIZE = 32

MODEL_PATH = f"models/densenet/densenet_model-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}.keras"

# Load data
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

# Display data shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Build model function with modifications
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), l2_reg=0.001):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-30:]:  # Adjust the number of layers to unfreeze as needed
        layer.trainable = True

    # layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),  # Increased units
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=1e-4)  # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )

    return model

if __name__ == "__main__":
    print("Building model...")
    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), l2_reg=0.001)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Learning rate scheduler using cosine annealing
    def cosine_annealing_schedule(epoch, lr):
        eta_min = 1e-6
        eta_max = 1e-3
        T_max = EPOCHS
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))

    lr_scheduler = LearningRateScheduler(cosine_annealing_schedule)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]
    )

    # Cleanup
    del model  # Deletes the model object
    del X_val, y_val  # Deletes dataset variables
    gc.collect()  # Forces garbage collection
    K.clear_session()