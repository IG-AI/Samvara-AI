# models/immaterial_layers.py

import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers

class ComplexLayer(layers.Layer):
    def __init__(self):
        super(ComplexLayer, self).__init__()

    def call(self, inputs):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        combined = tf.concat([real_part, imag_part], axis=-1)
        return combined

def quantum_circuit(inputs, weights):
    qml.Hadamard(wires=0)
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    return qml.expval(qml.PauliZ(0))

def build_immaterial_model():
    # Input placeholders for complex quantum data and material output
    quantum_input = layers.Input(shape=(2,), dtype='complex64', name='quantum_input')
    material_output = layers.Input(shape=(128,), name='material_output')

    # Quantum circuit layer
    weight_shapes = {"weights": 2}
    q_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=2, dtype='complex64')

    # Process quantum input
    quantum_output = q_layer(quantum_input)

    # Process complex numbers with custom layer
    complex_layer = ComplexLayer()(quantum_output)

    # Combine complex and material features
    combined = layers.Concatenate()([material_output, complex_layer])

    # Dense layer for final processing
    dense_layer = layers.Dense(128, activation='relu')(combined)

    # Build immaterial model
    immaterial_model = tf.keras.Model(inputs=[quantum_input, material_output], outputs=dense_layer)
    
    return immaterial_model
