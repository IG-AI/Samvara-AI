# models/samvara_model.py

from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model

def build_samvara_model():
    # Material model (image and text inputs)
    image_input = Input(shape=(32, 32, 3), name='image_input')
    text_input = Input(shape=(100,), name='text_input')
    material_model = build_material_model()

    material_output = material_model([image_input, text_input])

    # Immaterial model (quantum input)
    quantum_input = Input(shape=(2,), name='quantum_input')
    immaterial_model = build_immaterial_model()

    # Concatenate material and immaterial outputs
    immaterial_output = immaterial_model(quantum_input)

    combined_output = Concatenate()([material_output, immaterial_output])

    # Final dense layers for classification
    output = Dense(10, activation='softmax')(combined_output)

    model = Model(inputs=[image_input, text_input, quantum_input], outputs=output)
    return model
