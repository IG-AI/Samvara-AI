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
        self.real_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                           initializer='glorot_uniform',
                                           trainable=True,
                                           dtype=tf.float32)

        self.imaginary_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                                initializer='glorot_uniform',
                                                trainable=True,
                                                dtype=tf.float32)

    def call(self, inputs):
        real_input, imaginary_input = inputs

        # Perform matrix multiplication with the real and imaginary kernels
        real_output = tf.matmul(real_input, self.real_kernel)
        imaginary_output = tf.matmul(imaginary_input, self.imaginary_kernel)

        # Combine the real and imaginary outputs into complex numbers again
        output = tf.complex(real_output, imaginary_output)
        return output

    def compute_output_shape(self, input_shape):
        # Define the output shape of the layer
        return (input_shape[0], self.units)
    
    def compute_output_signature(self, input_signature):
        # Define the output dtype for the layer (complex64 in this case)
        return tf.TensorSpec(shape=self.compute_output_shape(input_signature.shape), dtype=tf.complex64)

# Build the immaterial model
def build_immaterial_model():
    real_input = layers.Input(shape=(2,), dtype=tf.float32, name="real_input")
    imaginary_input = layers.Input(shape=(2,), dtype=tf.float32, name="imaginary_input")

    # Use the custom quantum layer
    q_layer = CustomQuantumLayer(units=2, name="custom_quantum_layer")([real_input, imaginary_input])

    model = tf.keras.Model(inputs=[real_input, imaginary_input], outputs=q_layer)
    return model
