# models/material_layers.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, Embedding, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def build_material_model():
    # Image input (e.g., 32x32 images with 3 color channels)
    image_input = Input(shape=(32, 32, 3), name='image_input', dtype='float32')

    # Text input (e.g., a sequence of integers representing words)
    text_input = Input(shape=(100,), name='text_input', dtype='int32')

    # Image processing
    image_conv1 = Conv2D(32, (3, 3), activation='relu', name='conv1')(image_input)
    image_conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2')(image_conv1)
    image_flatten = Flatten()(image_conv2)
    image_dense = Dense(64, activation='relu', name='image_dense')(image_flatten)  # Reduced units to prevent overfitting
    image_dense = Dropout(0.5)(image_dense)  # Dropout for regularization
    image_dense = BatchNormalization()(image_dense)  # Batch Normalization

    # Text processing
    text_embed = Embedding(input_dim=10000, output_dim=64, name='text_embedding')(text_input)
    text_lstm = LSTM(64, name='lstm')(text_embed)  # Reduced units
    text_dense = Dense(64, activation='relu', name='text_dense')(text_lstm)
    text_dense = Dropout(0.5)(text_dense)  # Dropout for regularization
    text_dense = BatchNormalization()(text_dense)  # Batch Normalization

    # Concatenating both image and text representations
    material_concat = Concatenate(name='material_concat')([image_dense, text_dense])

    # Output layer for the material model
    material_output = Dense(128, activation='relu', name='material_output')(material_concat)

    # Creating the material model
    material_model = Model(inputs=[image_input, text_input], outputs=material_output, name='material_model')

    return material_model
