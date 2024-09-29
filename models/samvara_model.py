# samvara_model.py

from models.material_layers import build_hybrid_anastomatic_model
from models.immaterial_layers import build_immaterial_model
import tensorflow as tf
from tensorflow.keras import layers, models

def build_samvara_model():
    # Build material model (Layers 1-6)
    material_model = build_hybrid_anastomatic_model()

    # Build immaterial model (Layers 7-15)
    immaterial_model = build_immaterial_model()

    # Inputs for both models
    image_input = material_model.input[0]  # Image input from material model
    text_input = material_model.input[1]   # Text input from material model
    immaterial_feedback = material_model.input[2]  # Feedback loop input from immaterial layers

    # Quantum input for the immaterial model
    quantum_input = immaterial_model.input[0]  

    # Output from material decision layer (Layer 5 in material model)
    material_output = material_model.get_layer("dense_3").output

    # Feed material output (Layer 5) into immaterial model (starting from Layer 7)
    immaterial_output = immaterial_model([quantum_input, material_output])

    # Feed immaterial output (Layer 15) back into material model as feedback (Layer 1 refinement)
    final_output = material_model([image_input, text_input, immaterial_output])

    # Build the combined Samvara model
    samvara_model = models.Model(inputs=[image_input, text_input, quantum_input], outputs=final_output)

    # Compile the Samvara model
    samvara_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return samvara_model
