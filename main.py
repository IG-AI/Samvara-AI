# main.py

import os
import numpy as np
import tensorflow as tf
from models.samvara_model import build_samvara_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import logging

# Set logging configuration
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data Augmentation for Image Data
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load data
def load_data():
    num_samples = 1000
    image_data = np.random.random((num_samples, 32, 32, 3)).astype('float32')
    text_data = np.random.randint(10000, size=(num_samples, 100)).astype('int64')
    quantum_data_real = np.random.random((num_samples, 2)).astype('float32')  # Real part
    quantum_data_imaginary = np.random.random((num_samples, 2)).astype('float32')  # Imaginary part
    labels = np.random.randint(10, size=(num_samples, 10)).astype('int64')  # Assuming 10 classes
    return image_data, text_data, quantum_data_real, quantum_data_imaginary, labels

# Set up directories and ensure proper permissions
def ensure_directory_exists_and_writable(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logging.info(f"Directory {dir_path} exists and is writable.")

# Clear old checkpoints to prevent conflicts
def clear_existing_checkpoints(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".h5") or file.endswith(".weights.h5"):
            os.remove(os.path.join(checkpoint_dir, file))
    logging.info(f"Cleared existing checkpoints in {checkpoint_dir}.")

# Directory setup
checkpoint_dir = "checkpoints/"
ensure_directory_exists_and_writable(checkpoint_dir)
clear_existing_checkpoints(checkpoint_dir)

# Load data
image_data, text_data, quantum_real, quantum_imaginary, labels = load_data()

# Augment the image data
train_data_gen = data_augmentation.flow(image_data, labels, batch_size=BATCH_SIZE)

# Build the Samvara model
model = build_samvara_model()

# Compile the model with a lower learning rate for better optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Model Checkpoint: Save the best model during training with auto-generated filenames
checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model_{epoch:02d}.weights.h5'),
    save_best_only=True,
    monitor='val_loss',
    verbose=1,
    save_weights_only=True
)

# Early Stopping: Stop training if validation loss doesn't improve after 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Debugging and Logging
model.summary()
logging.info(f"Image data shape: {image_data.shape}, Text data shape: {text_data.shape}")
logging.info(f"Quantum data (real) shape: {quantum_real.shape}, Quantum data (imaginary) shape: {quantum_imaginary.shape}")

# Train the model
history = model.fit(
    [image_data, text_data, quantum_real, quantum_imaginary],
    labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save the model
model.save_weights(os.path.join(checkpoint_dir, 'final_model_weights.weights.h5'))
logging.info(f"Model weights saved.")
