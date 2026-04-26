import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\pc\OneDrive\Music\Desktop\ML_Projects\Dataset\cat_dog_dataset\train"
MODEL_SAVE  = "cat_dog_cnn.pth"
IMG_SIZE    = 128
BATCH_SIZE  = 64
EPOCHS      = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── Dataset & DataLoader ─────────────────────────────────────
dataset      = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
train_size   = int(0.85 * len(dataset))
val_size     = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

print(f"Classes : {dataset.class_to_idx}")
print(f"Train   : {train_size} | Val : {val_size}")

# ── CNN Model ────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 128 → 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 64 → 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)        # 32 → 16
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),  # 128×16×16 after 3 MaxPool
            nn.ReLU(),
            nn.Linear(256, 2)         # 2 classes: Cat, Dog
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)    # Flatten
        x = self.fc_layers(x)
        return x

model     = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# ── Training ─────────────────────────────────────────────────
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        output = model(xb)            # Forward Propagation
        loss   = criterion(output, yb)  # Loss
        loss.backward()               # BackPropagation
        optimizer.step()              # Update Params

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {epoch_loss:.4f}")

# ── Validation ───────────────────────────────────────────────
val_losses    = []
correct_labels = 0
total_labels   = 0

model.eval()

with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        output = model(xb)
        loss   = criterion(output, yb)
        val_losses.append(loss.item())

        _, predicted    = torch.max(output, 1)
        correct_labels += (predicted == yb).sum().item()
        total_labels   += yb.size(0)

print(f"\nAccuracy = {correct_labels / total_labels * 100:.2f}%")

# ── Save Model ───────────────────────────────────────────────
torch.save({
    "model_state_dict" : model.state_dict(),
    "class_to_idx"     : dataset.class_to_idx,
    "img_size"         : IMG_SIZE
}, MODEL_SAVE)

print(f"Model saved → {MODEL_SAVE}")
print(f"Class mapping: {dataset.class_to_idx}")
