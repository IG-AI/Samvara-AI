# models/samvara_model.py

import tensorflow as tf
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

    # Immaterial model (real and imaginary quantum inputs)
    real_input = Input(shape=(2,), name='real_input')
    imaginary_input = Input(shape=(2,), name='imaginary_input')
    immaterial_model = build_immaterial_model()

    immaterial_output = immaterial_model([real_input, imaginary_input])

    # Convert immaterial_output to float32 to match material_output type
    immaterial_output_real = tf.math.real(immaterial_output)
    immaterial_output_float = tf.cast(immaterial_output_real, dtype=tf.float32)

    # Concatenate material and immaterial outputs
    combined_output = Concatenate()([material_output, immaterial_output_float])

    # Final dense layers for classification
    output = Dense(10, activation='softmax')(combined_output)

    model = Model(inputs=[image_input, text_input, real_input, imaginary_input], outputs=output)
    return model
