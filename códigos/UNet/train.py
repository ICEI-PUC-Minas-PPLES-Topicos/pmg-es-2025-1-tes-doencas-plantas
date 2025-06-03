import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from unet import UNet

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
        return image, mask, img_name

# TRANSFORMAÇÕES
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# DATALOADER
dataset = LeafDataset(IMG_DIR, MASK_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODELO, PERDA, OTIMIZADOR
model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Função para calcular IoU com proteção contra divisão por zero
def iou_metric(preds, masks):
    preds = (preds > 0.5).float()
    masks = (masks > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    iou = torch.where(union == 0, torch.tensor(1.0, device=preds.device), intersection / union)
    return iou.mean().item()

# TREINAMENTO
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_iou = 0
    batches_validos = 0

    for batch_idx, (imgs, masks, _) in enumerate(loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, masks)
        total_loss += loss.item()

        iou = iou_metric(preds, masks)
        if not torch.isnan(torch.tensor(iou)):
            total_iou += iou
            batches_validos += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    iou_medio = total_iou / batches_validos if batches_validos > 0 else float("nan")
    print(f"Época {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f} - IoU médio: {iou_medio:.4f}")

    # Salva algumas predições visualmente
    model.eval()
    with torch.no_grad():
        for i in range(3):
            img, mask, nome = dataset[i]
            img_tensor = img.unsqueeze(0).to(DEVICE)
            pred = model(img_tensor).squeeze().cpu()
            save_image(pred, f"{SAVE_DIR}/pred_{epoch+1}_{nome}")
            save_image(mask, f"{SAVE_DIR}/real_{epoch+1}_{nome}")
