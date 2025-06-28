# build_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import gc
import tensorflow.keras.backend as K

# Disable ONEDNN for CPU compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMAGE_SIZE = 224
IMAGE_NUMBER = 8000
EPOCHS = 50
BATCH_SIZE = 64

model = "densenet"
model_name = f"{model}_model"

DATA_DIR = f"processed_data/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"
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
    inputs = Input(shape=input_shape)
    base_model = DenseNet121(weights="imagenet", include_top=False, input_tensor=inputs)
    
    # base_model.trainable = False  # Freeze all layers

    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-200:]:
        layer.trainable = True
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=3e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-4)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train model
print("Training model...")

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.02)
    image = tf.clip_by_value(image, 0.0, 1.0)
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