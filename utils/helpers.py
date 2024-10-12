# utils/helpers.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import h5py
import os
import logging

# Function to preprocess image data
def preprocess_images(image_data, image_size=(32, 32)):
    # Resize images to the target size
    image_data = np.array([tf.image.resize(image, image_size) for image in image_data])
    return image_data / 255.0  # Normalize images to [0,1]

# Function to preprocess text data
def preprocess_text(text_data, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

# Function to split dataset into train and test sets
def split_data(inputs, labels, test_size=0.2):
    return train_test_split(inputs, labels, test_size=test_size, random_state=42)

# Function to load data (placeholder, customize as needed)
def load_data(image_path, text_path, label_path):
    images = np.load(image_path)
    texts = np.load(text_path)
    labels = np.load(label_path)
    return images, texts, labels

# Ensure directory exists and is writable
def ensure_directory_exists_and_writable(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logging.info(f"Directory {dir_path} exists and is writable.")

# Clear existing checkpoints
def clear_existing_checkpoints(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".weights.h5") or file.endswith(".keras"):
            os.remove(os.path.join(checkpoint_dir, file))
    logging.info(f"Cleared existing checkpoints in {checkpoint_dir}.")

# Ensure that any pre-existing file is safely removed before creating a new one
def safe_remove(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        logging.info(f"Removed existing file: {filepath}")

# Ensure that existing datasets are removed in HDF5 files before creating a new one
def safe_remove_hdf5_dataset(filepath, dataset_name):
    if os.path.exists(filepath):
        with h5py.File(filepath, 'a') as h5file:
            if dataset_name in h5file:
                del h5file[dataset_name]
                logging.info(f"Removed existing dataset: {dataset_name} from {filepath}")

# Function to log GPU usage
def log_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

# Function to monitor GPU usage during training
def monitor_gpu_during_training(interval=60):
    while True:
        log_gpu_usage()
        time.sleep(interval)

# Function to start GPU monitoring in a separate thread
def start_gpu_monitoring(interval=60):
    gpu_monitor_thread = Thread(target=monitor_gpu_during_training, args=(interval,))
    gpu_monitor_thread.start()
    return gpu_monitor_thread
