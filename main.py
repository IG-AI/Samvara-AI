# main.py

import os
import numpy as np
import tensorflow as tf
from models.samvara_model import build_samvara_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py

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
    # Replace with actual data loading
    num_samples = 1000
    image_data = np.random.random((num_samples, 32, 32, 3))
    text_data = np.random.randint(10000, size=(num_samples, 100))
    quantum_data = np.random.random((num_samples, 2))
    labels = np.random.randint(10, size=(num_samples, 10))  # Assuming 10 classes
    
    return image_data, text_data, quantum_data, labels

# Load data
image_data, text_data, quantum_data, labels = load_data()

# Augment the image data
train_data_gen = data_augmentation.flow(image_data, labels, batch_size=BATCH_SIZE)

# Build the Samvara model
model = build_samvara_model()

# Compile the model with a lower learning rate for better optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define a function to safely remove an existing file
def safely_remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

# Create unique filename for checkpoints
checkpoint_path = 'best_model_{epoch:02d}-{val_loss:.2f}.h5'

# Safely remove the file if it exists before saving
def safe_model_checkpoint(filepath):
    # If file exists, remove it before saving new one
    safely_remove_file(filepath)

    # Return the ModelCheckpoint callback with the updated path
    return ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

# Model Checkpoint: Save the best model during training with unique filenames
checkpoint = safe_model_checkpoint(checkpoint_path)

# Early Stopping: Stop training if validation loss doesn't improve after 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    [image_data, text_data, quantum_data],
    labels,
    validation_split=0.2,  # Use 20% of the data for validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate([image_data, text_data, quantum_data], labels)
print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")
