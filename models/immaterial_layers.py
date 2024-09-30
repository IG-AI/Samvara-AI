# models/immaterial_layers.py

import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Complex Layer to handle complex numbers
class ComplexLayer(Layer):
    def __init__(self, units=32, input_dim=32):
        super(ComplexLayer, self).__init__()
        self.units = units
        self.input_dim = input_dim

        # Initialize real and imaginary parts separately
        self.real_weights = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer='random_normal',
                                            trainable=True)
        self.imag_weights = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer='random_normal',
                                            trainable=True)

    def call(self, inputs):
        real_part = tf.matmul(tf.math.real(inputs), self.real_weights)
        imag_part = tf.matmul(tf.math.imag(inputs), self.imag_weights)
        return tf.complex(real_part, imag_part)

# Define quantum circuit
def quantum_circuit(inputs, weights):
    qml.RX(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Create the QNode
dev = qml.device('default.qubit', wires=2)
qnode = qml.QNode(quantum_circuit, dev)

def quantum_layer():
    weight_shapes = {"weights": (2,)}
    return qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2, dtype='complex64')

# Build immaterial model
def build_immaterial_model():
    quantum_input = tf.keras.Input(shape=(2,), dtype=tf.complex64, name="quantum_input")
    
    # Use ComplexLayer to handle complex numbers
    complex_layer = ComplexLayer(units=64, input_dim=2)
    processed_complex = complex_layer(quantum_input)

    quantum_output = quantum_layer()(processed_complex)
    immaterial_model = tf.keras.Model(inputs=quantum_input, outputs=quantum_output)
    return immaterial_model
