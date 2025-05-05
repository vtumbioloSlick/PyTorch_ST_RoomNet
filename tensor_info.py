import torch

data = torch.load("cached_train/00afbdb9017087758fabd257476b3d508122d9e4.pt")

# Check what kind of object it is
print(type(data))

# If it's a dict, list keys
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k in data:
        print(f"{k}: {type(data[k])}, shape: {getattr(data[k], 'shape', 'N/A')}")
elif isinstance(data, (list, tuple)):
    for i, item in enumerate(data):
        print(f"Item {i}: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
else:
    print("Unknown format")



