import torch
from full_model import FullModel  # Make sure this matches the training model

# Load the checkpoint file
checkpoint = torch.load("model_epoch_150.pt", map_location='cpu')

# Rebuild the model architecture
model = FullModel()
model.load_state_dict(checkpoint['model_state_dict'])

# Optional: print model structure
print(model)
