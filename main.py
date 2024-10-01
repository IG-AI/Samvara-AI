# main.py

import os
import numpy as np
import tensorflow as tf
from models.samvara_model import build_samvara_model
from evolutionary_algorithm import run_evolutionary_algorithm  # New import for evolutionary algorithms
from reinforcement_learning import MentorModel  # New import for mentor reinforcement
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# Set logging configuration
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001  # Adjusted learning rate

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

    # Normalize the image, text, and quantum data to the 0-1 range
    image_data = image_data / 255.0  # Normalize image data
    text_data = text_data / 9999.0  # Normalize text data to [0, 1]
    quantum_data_real = quantum_data_real / np.max(quantum_data_real)  # Normalize quantum data
    quantum_data_imaginary = quantum_data_imaginary / np.max(quantum_data_imaginary)

    return image_data, text_data, quantum_data_real, quantum_data_imaginary, labels

# Directory setup and checkpoint management
def ensure_directory_exists_and_writable(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logging.info(f"Directory {dir_path} exists and is writable.")

def clear_existing_checkpoints(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".h5") or file.endswith(".weights.h5"):
            os.remove(os.path.join(checkpoint_dir, file))
    logging.info(f"Cleared existing checkpoints in {checkpoint_dir}.")

# Directory for checkpoints
checkpoint_dir = "checkpoints/"
ensure_directory_exists_and_writable(checkpoint_dir)
clear_existing_checkpoints(checkpoint_dir)

# Load data
image_data, text_data, quantum_real, quantum_imaginary, labels = load_data()

# Step 1: Train Material Layers (Subconscious Development)
logging.info("Starting Phase 1: Training Material Layers (Subconscious Development)")

# Build the Samvara model with material layers only
material_model = build_samvara_model(include_immaterial=False)  # Train without immaterial layers

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
material_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the material model
material_history = material_model.fit(
    [image_data, text_data], labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'material_best_model.h5'), save_best_only=True)],
    verbose=1
)

# Save weights for material layers (Subconscious)
material_model.save_weights(os.path.join(checkpoint_dir, 'material_final_model_weights.h5'))

# Step 2: Evolve Microbiome Model (Evolutionary Algorithm)
logging.info("Running Evolutionary Algorithm to simulate Microbiome Influence")

evolutionary_data = run_evolutionary_algorithm(image_data, text_data, quantum_real, quantum_imaginary, labels)

# Step 3: Train Immaterial Layers (Conscious Development)
logging.info("Starting Phase 2: Training Immaterial Layers (Conscious Development)")

# Build the full Samvara model (material + immaterial layers)
samvara_model = build_samvara_model(include_immaterial=True)

# Compile the full model
samvara_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Introduce mentor-based reinforcement learning
mentor = MentorModel()

# Train the full model (conscious + evolved microbiome data)
full_history = samvara_model.fit(
    [image_data, text_data, quantum_real, quantum_imaginary], labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[mentor, EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'samvara_best_model.h5'), save_best_only=True)],
    verbose=1
)

# Save the final model
samvara_model.save_weights(os.path.join(checkpoint_dir, 'samvara_final_model_weights.h5'))
logging.info(f"Final Samvara Model weights saved.")
