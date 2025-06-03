import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 32x32
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),  # 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),   # 128x128
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2), nn.Sigmoid()   # 256x256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
