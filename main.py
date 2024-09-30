# main.py

import os
import numpy as np
import tensorflow as tf
from models.samvara_model import build_samvara_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import time
import uuid
import stat
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

# Efficient data handling (dummy data for example purposes)
def load_data():
    num_samples = 1000
    image_data = np.random.random((num_samples, 32, 32, 3)).astype('float32')
    text_data = np.random.randint(10000, size=(num_samples, 100)).astype('int64')
    quantum_data_real = np.random.random((num_samples, 2)).astype('float32')  # Real part
    quantum_data_imaginary = np.random.random((num_samples, 2)).astype('float32')  # Imaginary part
    labels = np.random.randint(10, size=(num_samples, 10)).astype('int64')  # Assuming 10 classes
    return image_data, text_data, quantum_data_real, quantum_data_imaginary, labels

# Function to set directory permissions and create the directory if it doesn't exist
def ensure_directory_exists_and_writable(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Make sure directory is writable
    os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Set 777 permissions
    logging.info(f"Directory {dir_path} exists and is writable.")

# Clear old checkpoints to prevent conflicts
def clear_existing_checkpoints(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".h5") or file.endswith(".weights.h5"):
            os.remove(os.path.join(checkpoint_dir, file))
    logging.info(f"Cleared existing checkpoints in {checkpoint_dir}.")

# Set up directories and ensure proper permissions
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
    filepath=os.path.join(checkpoint_dir, 'best_model_{epoch:02d}.weights.h5'),  # Ensure the file ends with `.weights.h5`
    save_best_only=True,
    monitor='val_loss',
    verbose=1,
    save_weights_only=True  # Save only weights
)

# Early Stopping: Stop training if validation loss doesn't improve after 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Print the model summary to inspect the layers
model.summary()

# Train the model (passing quantum_real and quantum_imaginary separately)
history = model.fit(
    [image_data, text_data, quantum_real, quantum_imaginary],  # Pass real and imaginary inputs separately
    labels,
    validation_split=0.2,  # Use 20% of the data for validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Manually save the model's weights after training
final_weights_path = os.path.join(checkpoint_dir, 'final_model_weights.weights.h5')
model.save_weights(final_weights_path)
logging.info(f"Weights saved to {final_weights_path}")

# Manually save the entire model after training
final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
model.save(final_model_path)
logging.info(f"Model saved to {final_model_path}")

# Evaluate the model
loss, accuracy = model.evaluate([image_data, text_data, quantum_real, quantum_imaginary], labels)
logging.info(f"Final Loss: {loss}, Final Accuracy: {accuracy}")
