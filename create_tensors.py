import os
import torch
import numpy as np
from PIL import Image
import scipy.io


input_dir = 'surface_relabel/train'
output_dir = 'cached_train'
mat_key = "layout"

os.makedirs(output_dir, exist_ok=True)

# process the files into a tensor
def load_triplet(base_path):
    img = Image.open(base_path + ".jpg").convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    seg = Image.open(base_path + ".png")
    seg_tensor = torch.from_numpy(np.array(seg)).long()

    mat = scipy.io.loadmat(base_path + ".mat")
    ref_tensor = torch.from_numpy(mat[mat_key]).long() - 1

    return {"img": img_tensor, "seg": seg_tensor, "ref": ref_tensor}

# search for files with the same name but different extensions
for file in os.listdir(input_dir):
    if file.endswith(".jpg"):
        base = os.path.splitext(file)[0]
        jpg_path = os.path.join(input_dir, base + ".jpg")
        png_path = os.path.join(input_dir, base + ".png")
        mat_path = os.path.join(input_dir, base + ".mat")

        if os.path.exists(png_path) and os.path.exists(mat_path):
            try:
                data = load_triplet(os.path.join(input_dir, base))
                torch.save(data, os.path.join(output_dir, base + ".pt"))
                print(f"Saved: {base}.pt")
            except Exception as e:
                print(f"Failed to process {base}: {e}")
