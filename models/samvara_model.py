# models/samvara_model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer
from tensorflow.keras.models import Model
from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model

class RealPartLayer(Layer):
    def call(self, inputs):
        return inputs  # Removed real extraction; both parts are treated as float32.

def build_samvara_model():
    # Material model
    image_input = Input(shape=(32, 32, 3), name='image_input')
    text_input = Input(shape=(100,), name='text_input')
    material_model = build_material_model()
    material_output = material_model([image_input, text_input])

    # Immaterial model
    real_input = Input(shape=(2,), name='real_input')
    imaginary_input = Input(shape=(2,), name='imaginary_input')
    immaterial_model = build_immaterial_model()
    immaterial_output = immaterial_model([real_input, imaginary_input])

    combined_output = Concatenate()([material_output, immaterial_output])

    # Final dense layer
    output = Dense(10, activation='softmax')(combined_output)

    model = Model(inputs=[image_input, text_input, real_input, imaginary_input], outputs=output)
    return model
