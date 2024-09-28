# main.py

from models.material_layers import build_material_model
from models.immaterial_layers import build_immaterial_model

# Build the material model (Layers 1-6)
material_model = build_material_model()

# Build the immaterial model (Layers 7-15)
immaterial_model = build_immaterial_model()

# Optionally: Combine both models (if you want them as one system)
# For example, you could connect the output of material_model to the input of immaterial_model
# and create a unified model.

# Summarize the models
print("Material Model Summary:")
material_model.summary()

print("\nImmaterial Model Summary:")
immaterial_model.summary()
