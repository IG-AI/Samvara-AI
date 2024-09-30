# models/immaterial_layers.py

import pennylane as qml
from tensorflow.keras.layers import Layer
import tensorflow as tf
from pennylane import numpy as np
from pennylane.qnn import KerasLayer

class ComplexLayer(Layer):
    def __init__(self, units, **kwargs):
        super(ComplexLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.real_kernel = self.add_weight(
            name="real_kernel",
            shape=(input_shape[-1], self.units),
            initializer="uniform",
            dtype="float32",
            trainable=True,
        )
        self.imag_kernel = self.add_weight(
            name="imag_kernel",
            shape=(input_shape[-1], self.units),
            initializer="uniform",
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        real_output = tf.matmul(tf.math.real(inputs), self.real_kernel)
        imag_output = tf.matmul(tf.math.imag(inputs), self.imag_kernel)
        return tf.complex(real_output, imag_output)

def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(2))
    qml.templates.BasicEntanglerLayers(weights, wires=range(2))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

def build_immaterial_model():
    # Quantum layer with complex numbers support
    n_qubits = 2
    weight_shapes = {"weights": (3, n_qubits)}
    
    # Using Pennylane KerasLayer for quantum computations
    q_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=2, dtype="complex64")

    # Custom ComplexLayer to handle complex data
    complex_layer = ComplexLayer(10)  # Example: 10 units in the complex layer

    # Inputs for the immaterial model
    quantum_input = tf.keras.Input(shape=(2,), dtype="complex64")

    # Apply quantum layer and complex layer
    quantum_output = q_layer(quantum_input)
    complex_output = complex_layer(quantum_output)

    # Create the immaterial model
    immaterial_model = tf.keras.Model(inputs=quantum_input, outputs=complex_output)
    return immaterial_model
