# models/samvara_model.py

from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model
from tensorflow.keras import layers, models
import tensorflow as tf

def build_samvara_model():
    material_input_image = layers.Input(shape=(32, 32, 3), name='material_input_image')
    material_input_text = layers.Input(shape=(100,), name='material_input_text')
    quantum_input = layers.Input(shape=(2,), name='quantum_input', dtype=tf.complex64)

    material_model = build_material_model()
    material_output = material_model([material_input_image, material_input_text])

    immaterial_model = build_immaterial_model()
    immaterial_output = immaterial_model([quantum_input, material_output])

    # Split real and imaginary parts of the immaterial output
    real_part = tf.math.real(immaterial_output)
    imag_part = tf.math.imag(immaterial_output)

    # Combine the real and imaginary parts with the material output
    combined_output = layers.Concatenate()([material_output, real_part, imag_part])

    final_output = layers.Dense(10, activation='softmax')(combined_output)

    return models.Model(inputs=[material_input_image, material_input_text, quantum_input], outputs=final_output)
