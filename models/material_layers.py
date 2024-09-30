# models/material_layers.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, Embedding, Concatenate, Dropout
from tensorflow.keras.models import Model

def build_material_model():
    # Image input (e.g., 32x32 images with 3 color channels)
    image_input = Input(shape=(32, 32, 3), name='image_input', dtype='float32')

    # Text input (e.g., a sequence of integers representing words)
    text_input = Input(shape=(100,), name='text_input', dtype='int32')

    # Image processing
    image_conv1 = Conv2D(32, (3, 3), activation='relu', name='conv1')(image_input)  # Convolutional layer for feature extraction
    image_conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2')(image_conv1)  # Deeper convolutional layer
    image_flatten = Flatten()(image_conv2)
    image_dense = Dense(128, activation='relu', name='image_dense')(image_flatten)

    # Text processing
    text_embed = Embedding(input_dim=10000, output_dim=64, name='text_embedding')(text_input)  # Embedding for text sequences
    text_lstm = LSTM(128, name='lstm')(text_embed)  # LSTM layer for sequence processing
    text_dense = Dense(128, activation='relu', name='text_dense')(text_lstm)

    # Concatenating both image and text representations
    material_concat = Concatenate(name='material_concat')([image_dense, text_dense])

    # Output layer for the material model
    material_output = Dense(128, activation='relu', name='material_output')(material_concat)

    # Creating the material model
    material_model = Model(inputs=[image_input, text_input], outputs=material_output, name='material_model')

    return material_model
