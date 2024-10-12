# utils/helpers.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import h5py
import os

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
    # Implement data loading logic here
    images = np.load(image_path)
    texts = np.load(text_path)
    labels = np.load(label_path)
    
    return images, texts, labels
