# models/samvara_model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer
from tensorflow.keras.models import Model
from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model

def build_samvara_model(include_immaterial=True):
    """
    Build the Samvara model. If include_immaterial=False, only train material layers.
    """
    # Material model (image and text inputs)
    image_input = Input(shape=(32, 32, 3), name='image_input')
    text_input = Input(shape=(100,), name='text_input')
    material_model = build_material_model()

    material_output = material_model([image_input, text_input])

    if include_immaterial:
        # Immaterial model (quantum input)
        real_input = Input(shape=(2,), name='real_input')
        imaginary_input = Input(shape=(2,), name='imaginary_input')
        immaterial_model = build_immaterial_model()

        immaterial_output = immaterial_model([real_input, imaginary_input])

        # Concatenate material and immaterial outputs
        combined_output = Concatenate()([material_output, immaterial_output])
    else:
        # Only material output
        combined_output = material_output

    # Final dense layer for classification
    output = Dense(10, activation='softmax')(combined_output)

    if include_immaterial:
        return Model(inputs=[image_input, text_input, real_input, imaginary_input], outputs=output)
    else:
        return Model(inputs=[image_input, text_input], outputs=output)
