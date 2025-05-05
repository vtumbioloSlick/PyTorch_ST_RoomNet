import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Full Model Currently Using Projective Transformer
from full_model import FullModel


# Cached dataset class
class CachedSurfaceDataset(Dataset):
    def __init__(self, cached_dir):
        self.cached_dir = cached_dir
        self.files = sorted([f for f in os.listdir(cached_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(os.path.join(self.cached_dir, self.files[idx]))
        return sample['img'], sample['seg'], sample['ref']


if __name__ == '__main__':
    print("1")
    # Load datasets
    train_dataset = CachedSurfaceDataset('cached_train')
    val_dataset = CachedSurfaceDataset('cached_val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print("1")
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullModel().to(device)

    # Huber Loss
    criterion = nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Matching the 150 epocs used in tensorflow implementation
    num_epochs = 150
    best_val_loss = float('inf')
    print("1")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, segs, ref_imgs in train_loader:
            imgs, segs, ref_imgs = imgs.to(device), segs.to(device), ref_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, ref_imgs)

            loss = criterion(outputs.squeeze(1), segs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("1")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # val data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, segs, ref_imgs in val_loader:
                imgs, segs, ref_imgs = imgs.to(device), segs.to(device), ref_imgs.to(device)

                outputs = model(imgs, ref_imgs)
                loss = criterion(outputs.squeeze(1), segs)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"✅ Saved new best model at epoch {epoch + 1}")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'model_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")

