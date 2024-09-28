# models/material_layers.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_material_model():
    # Input layer for images (32x32 RGB images)
    image_input = layers.Input(shape=(32, 32, 3), name='image_input')

    # Input layer for text (sequence of word embeddings)
    text_input = layers.Input(shape=(100,), name='text_input')

    # Image processing
    image_features = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
    image_features = layers.Conv2D(64, (3, 3), activation='relu')(image_features)

    # Text processing
    embedding_layer = layers.Embedding(input_dim=10000, output_dim=128, input_length=100)(text_input)
    text_features = layers.LSTM(128)(embedding_layer)

    # Combine both image and text features
    image_features_flat = layers.Flatten()(image_features)
    combined_features = layers.Concatenate()([image_features_flat, text_features])

    # Dense layers for mid-level processing
    dense_layer = layers.Dense(512, activation='relu')(combined_features)
    dense_layer = layers.Dropout(0.5)(dense_layer)

    # Contextual understanding
    contextual_layer = layers.Dense(256, activation='relu')(dense_layer)

    # Decision making
    decision_layer = layers.Dense(128, activation='relu')(contextual_layer)

    # Feedback and refinement
    refinement_layer = layers.Dense(64, activation='relu')(decision_layer)

    # Output layer
    output = layers.Dense(10, activation='softmax')(refinement_layer)

    # Build the model
    model = models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
