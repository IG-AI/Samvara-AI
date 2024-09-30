# models/samvara_model.py

from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model
from tensorflow.keras import layers, models
import tensorflow as tf

def build_samvara_model():
    # Input placeholders
    material_input_image = layers.Input(shape=(32, 32, 3), name='material_input_image')
    material_input_text = layers.Input(shape=(100,), name='material_input_text')
    quantum_input = layers.Input(shape=(2,), name='quantum_input', dtype=tf.complex64)  # Set quantum input to complex

    # Material model (image and text processing)
    material_model = build_material_model()
    material_output = material_model([material_input_image, material_input_text])

    # Immaterial model (quantum processing)
    immaterial_model = build_immaterial_model()
    immaterial_output = immaterial_model([quantum_input, material_output])

    # Split real and imaginary parts from the immaterial output
    real_part = tf.math.real(immaterial_output)
    imag_part = tf.math.imag(immaterial_output)

    # Combine material output with real and imaginary parts of immaterial output
    combined_output = layers.Concatenate()([material_output, real_part, imag_part])

    # Final output layer (softmax classification)
    final_output = layers.Dense(10, activation='softmax')(combined_output)

    # Build the Samvara model
    model = models.Model(inputs=[material_input_image, material_input_text, quantum_input], outputs=final_output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
