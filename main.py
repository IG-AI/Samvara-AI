# main.py

import time
import os
import numpy as np
import tensorflow as tf
from models.samvara_model import build_samvara_model
from models.microbiome_model import run_evolutionary_algorithm
from models.mentor_model import MentorModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from utils.helpers import safe_remove, safe_remove_hdf5_dataset, ensure_directory_exists_and_writable, clear_existing_checkpoints

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
    quantum_data_real = np.random.random((num_samples, 2)).astype('float32')
    quantum_data_imaginary = np.random.random((num_samples, 2)).astype('float32')
    labels = np.random.randint(10, size=(num_samples, 10)).astype('int64')

    # Normalize the image, text, and quantum data
    image_data = image_data / 255.0
    text_data = text_data / 9999.0
    quantum_data_real = quantum_data_real / np.max(quantum_data_real)
    quantum_data_imaginary = quantum_data_imaginary / np.max(quantum_data_imaginary)

    return image_data, text_data, quantum_data_real, quantum_data_imaginary, labels

# Directory for checkpoints
checkpoint_dir = "checkpoints/"
ensure_directory_exists_and_writable(checkpoint_dir)
clear_existing_checkpoints(checkpoint_dir)

# Load data
image_data, text_data, quantum_real, quantum_imaginary, labels = load_data()

# Step 1: Train Material Layers (Subconscious Development)
logging.info("Starting Phase 1: Training Material Layers (Subconscious Development)")

# Build the Samvara model with material layers only
material_model = build_samvara_model(include_immaterial=False)

# Compile the material model
material_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
material_model.compile(optimizer=material_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Use a timestamp for unique file names to avoid conflicts
timestamp = int(time.time())

# Define filepaths
material_best_model_path = os.path.join(checkpoint_dir, f'material_best_model_{timestamp}.keras')
material_final_weights_path = os.path.join(checkpoint_dir, f'material_final_model_weights_{timestamp}.weights.h5')

# Ensure pre-existing files or datasets are safely removed
safe_remove(material_best_model_path)
safe_remove(material_final_weights_path)

# Train and save the best model during training
material_history = material_model.fit(
    [image_data, text_data], labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               ModelCheckpoint(filepath=material_best_model_path, save_best_only=True)],
    verbose=1
)

# Save only the final weights with a .weights.h5 extension
material_model.save_weights(material_final_weights_path)

# Step 2: Evolve Microbiome Model (Evolutionary Algorithm)
logging.info("Running Evolutionary Algorithm to simulate Microbiome Influence")
evolutionary_data = run_evolutionary_algorithm(image_data, text_data, quantum_real, quantum_imaginary, labels)

# Step 3: Train Immaterial Layers (Conscious Development)
logging.info("Starting Phase 2: Training Immaterial Layers (Conscious Development)")

# Build the full Samvara model (material + immaterial layers)
samvara_model = build_samvara_model(include_immaterial=True)

# Compile the full model
full_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
samvara_model.compile(optimizer=full_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define filepaths for the full model
samvara_best_model_path = os.path.join(checkpoint_dir, f'samvara_best_model_{timestamp}.keras')
samvara_final_model_path = os.path.join(checkpoint_dir, f'samvara_final_model_{timestamp}.keras')

# Ensure pre-existing files or datasets are safely removed
safe_remove(samvara_best_model_path)
safe_remove(samvara_final_model_path)

# Introduce mentor-based reinforcement learning
mentor = MentorModel()

full_history = samvara_model.fit(
    [image_data, text_data, quantum_real, quantum_imaginary], labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[mentor, EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               ModelCheckpoint(filepath=samvara_best_model_path, save_best_only=True)],
    verbose=1
)

# Ensure no dataset conflicts in the HDF5 file before saving the final model
safe_remove_hdf5_dataset(samvara_final_model_path, 'model_weights')

# Save the full model (architecture + weights) using .keras
samvara_model.save(samvara_final_model_path)
logging.info(f"Final Samvara Model saved.")
