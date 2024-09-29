# models/samvara_model.py

from tensorflow.keras import layers, models
from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model

def build_samvara_model():
    # Material model
    material_input = layers.Input(shape=(32, 32, 3), name='material_input')
    material_output = build_material_model()(material_input)

    # Quantum input
    quantum_input = layers.Input(shape=(2,), name='quantum_input')

    # Concatenate quantum and material outputs
    concatenated_input = layers.Concatenate()([quantum_input, material_output])

    # Immaterial model (accepting the concatenated input)
    immaterial_model = build_immaterial_model()
    
    # Ensure we are passing the concatenated input to the immaterial model
    immaterial_output = immaterial_model(concatenated_input)

    # Build the full Samvara model
    model = models.Model(inputs=[material_input, quantum_input], outputs=immaterial_output, name='Samvara_Model')

    return model
