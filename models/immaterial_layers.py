# models/immaterial_layers.py

import tensorflow as tf
from tensorflow.keras import layers

# Custom Quantum Layer
class CustomQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=2, **kwargs):
        super(CustomQuantumLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize real and imaginary parts for the complex weights
        self.real_kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                           initializer='glorot_uniform',
                                           trainable=True,
                                           dtype=tf.float32)

        self.imaginary_kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                                initializer='glorot_uniform',
                                                trainable=True,
                                                dtype=tf.float32)

    def call(self, inputs):
        # Expecting inputs as a list of real and imaginary parts
        real_part, imaginary_part = inputs

        # Perform matrix multiplication with the real and imaginary kernels
        real_output = tf.matmul(real_part, self.real_kernel)
        imaginary_output = tf.matmul(imaginary_part, self.imaginary_kernel)

        # Combine the real and imaginary outputs into complex numbers again
        output = tf.complex(real_output, imaginary_output)
        return output

# Build the immaterial model
def build_immaterial_model():
    real_input = layers.Input(shape=(2,), dtype=tf.float32, name="real_input")
    imaginary_input = layers.Input(shape=(2,), dtype=tf.float32, name="imaginary_input")

    q_layer = CustomQuantumLayer(units=2, name="custom_quantum_layer")([real_input, imaginary_input])

    model = tf.keras.Model(inputs=[real_input, imaginary_input], outputs=q_layer)
    return model
