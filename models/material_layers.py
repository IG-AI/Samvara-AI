# models/material_layers.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_material_model():
    # Input layer for images and text
    image_input = layers.Input(shape=(32, 32, 3), name='image_input')
    text_input = layers.Input(shape=(100,), name='text_input')

    # Image processing (Layer 1)
    image_features = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    image_features = layers.MaxPooling2D(pool_size=(2, 2))(image_features)
    image_features = layers.Conv2D(64, (3, 3), activation='relu')(image_features)

    # Text processing (Layer 1)
    embedding_layer = layers.Embedding(input_dim=10000, output_dim=128, input_length=100)(text_input)
    text_features = layers.LSTM(128)(embedding_layer)

    # Combine both image and text features (Layer 2)
    image_features_flat = layers.Flatten()(image_features)
    combined_features = layers.Concatenate()([image_features_flat, text_features])
    dense_layer = layers.Dense(512, activation='relu')(combined_features)

    # Lateral connection between Layer 3 and Layer 4
    dense_layer = layers.Dropout(0.5)(dense_layer)
    contextual_layer = layers.Dense(256, activation='relu')(dense_layer)
    lateral_combined = layers.Concatenate()([dense_layer, contextual_layer])
    lateral_layer = layers.Dense(256, activation='relu')(lateral_combined)

    # Decision making (Layer 5)
    decision_layer = layers.Dense(128, activation='relu')(lateral_layer)

    # Feedback from immaterial layer (Layer 15) to refine sensory input
    immaterial_feedback = layers.Input(shape=(64,), name='immaterial_feedback')
    refined_input = layers.Concatenate()([decision_layer, immaterial_feedback])
    refined_layer = layers.Dense(128, activation='relu')(refined_input)

    # Final refinement (Layer 6)
    refinement_layer = layers.Dense(64, activation='relu')(refined_layer)

    # Output layer
    output = layers.Dense(10, activation='softmax')(refinement_layer)

    # Build the model
    model = models.Model(inputs=[image_input, text_input, immaterial_feedback], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
