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
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    weight_shapes = {"weights": (6,)}
    
    # Explicitly set the initializer for complex weights
    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
    
    # Ensure complex64 type for the weights
    q_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=2, kernel_initializer=kernel_initializer, dtype='complex64')
    
    return q_layer(inputs)

def build_immaterial_model():
    # Inputs: quantum data and material model output
    quantum_input = layers.Input(shape=(2,), name='quantum_input')
    material_output = layers.Input(shape=(128,), name='material_output')  # Shape from the material model

    # Quantum layer processing
    quantum_output = quantum_layer(quantum_input)

    # Split real and imaginary parts of quantum output
    real_part = tf.math.real(quantum_output)
    imaginary_part = tf.math.imag(quantum_output)

    # Dense layers processing immaterial inputs
    real_dense = layers.Dense(64, activation='relu')(real_part)
    imag_dense = layers.Dense(64, activation='relu')(imaginary_part)

    # Combine real and imaginary processed outputs
    combined_quantum = layers.Concatenate()([real_dense, imag_dense])

    # Further dense layers for immaterial processing
    immaterial_dense = layers.Dense(128, activation='relu')(combined_quantum)

    # Combine immaterial with material output
    combined = layers.Concatenate()([material_output, immaterial_dense])

    # Final dense layer to produce the immaterial output
    output = layers.Dense(128, activation='relu')(combined)

    # Build the immaterial model
    model = models.Model(inputs=[quantum_input, material_output], outputs=output)

    return model
