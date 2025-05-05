import torch
import matplotlib.pyplot as plt
from full_model import FullModel

# === CONFIG ===
model_path = "model_epoch_150.pt"
sample_path = "cached_train/0fd02506af4f0490759a528643993c1bc311f15c.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load sample ===
sample = torch.load(sample_path)
img = sample['img'].unsqueeze(0).to(device)
ref = sample['ref'].unsqueeze(0).to(device)

# === Load model ===
model = FullModel().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Predict ===
with torch.no_grad():
    pred = model(img, ref).squeeze().cpu()

# === Show prediction only ===
plt.imshow(pred.clamp(0, 5), cmap='gray')  # clamp if needed
plt.title("Model Prediction")
plt.axis("off")
plt.show()
