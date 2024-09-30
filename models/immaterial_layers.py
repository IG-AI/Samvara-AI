# models/immaterial_layers.py

import tensorflow as tf
from tensorflow.keras import layers, models
import pennylane as qml

# Quantum layer using PennyLane
def quantum_layer(inputs):
    # Define a PennyLane device
    dev = qml.device('default.qubit', wires=2)

    # Define the quantum node (quantum circuit)
    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.RX(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Rot(weights[0], weights[1], weights[2], wires=0)
        qml.Rot(weights[3], weights[4], weights[5], wires=1)
        return [qml.expval(qml.PauliZ(i)) + 1j * qml.expval(qml.PauliX(i)) for i in range(2)]

    weight_shapes = {"weights": (6,)}
    q_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=2, dtype=tf.complex64)

    return q_layer(inputs)

def build_immaterial_model():
    quantum_input = layers.Input(shape=(2,), dtype=tf.complex64, name='quantum_input')
    material_output = layers.Input(shape=(128,), name='material_output')

    quantum_output = quantum_layer(quantum_input)

    # Split real and imaginary parts
    real_part = tf.math.real(quantum_output)
    imaginary_part = tf.math.imag(quantum_output)

    # Process real and imaginary parts separately
    real_dense = layers.Dense(64, activation='relu')(real_part)
    imag_dense = layers.Dense(64, activation='relu')(imaginary_part)

    # Combine real and imaginary parts
    combined_quantum = layers.Concatenate()([real_dense, imag_dense])

    immaterial_dense = layers.Dense(128, activation='relu')(combined_quantum)
    combined = layers.Concatenate()([material_output, immaterial_dense])

    output = layers.Dense(128, activation='relu')(combined)

    return models.Model(inputs=[quantum_input, material_output], outputs=output)
