# models/material_layers.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model

def build_material_model():
    # Image input (e.g., 32x32 images with 3 color channels)
    image_input = Input(shape=(32, 32, 3), name='image_input', dtype='float32')
    
    # Text input (e.g., a sequence of integers representing words)
    text_input = Input(shape=(100,), name='text_input', dtype='int32')
    
    # Image processing
    image_flatten = tf.keras.layers.Flatten()(image_input)
    image_dense = Dense(128, activation='relu', name='image_dense')(image_flatten)
    
    # Text processing (e.g., embedding + dense layer)
    text_embed = tf.keras.layers.Embedding(input_dim=10000, output_dim=64, name='text_embedding')(text_input)
    text_flatten = tf.keras.layers.Flatten()(text_embed)
    text_dense = Dense(128, activation='relu', name='text_dense')(text_flatten)

    # Concatenating both image and text representations
    material_concat = Concatenate(name='material_concat')([image_dense, text_dense])
    
    # Output layer for the material model
    material_output = Dense(128, activation='relu', name='material_output')(material_concat)
    
    # Creating the material model
    material_model = Model(inputs=[image_input, text_input], outputs=material_output, name='material_model')
    
    return material_model
