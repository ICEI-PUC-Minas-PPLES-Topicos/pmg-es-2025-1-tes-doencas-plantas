import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from resunet import ResUNet

# CONFIGURAÇÕES
IMG_DIR = "imagens"
MASK_DIR = "mascaras"
SAVE_DIR = "resultados"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# DATASET
class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir)
                       if os.path.exists(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask, img_name  # inclui nome da imagem

# TRANSFORMAÇÕES
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# DATALOADER
dataset = LeafDataset(IMG_DIR, MASK_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODELO, PERDA, OTIMIZADOR
model = ResUNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# FUNÇÃO IoU
def iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return (intersection / union).mean().item()

# TREINAMENTO
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_iou = 0

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)
        total_loss += loss.item()
        total_iou += iou(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    print(f"Época {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - IoU médio: {avg_iou:.4f}")

    # Salva algumas predições visualmente
    model.eval()
    with torch.no_grad():
        for i in range(3):
            img, mask, img_name = dataset[i]
            img = img.unsqueeze(0).to(DEVICE)
            pred = model(img).squeeze().cpu()

            suffix = os.path.splitext(img_name)[0][-4:]  # últimos 4 caracteres
            save_image(pred, f"{SAVE_DIR}/pred_{epoch+1}_{suffix}.png")
            save_image(mask, f"{SAVE_DIR}/real_{epoch+1}_{suffix}.png")
