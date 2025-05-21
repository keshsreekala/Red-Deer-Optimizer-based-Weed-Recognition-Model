#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import random

# ==== Custom CBAM Module ====
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)

# ==== RDODL-WRMAIR Model ====
class RDODL_WRMAIR(nn.Module):
    def __init__(self):
        super(RDODL_WRMAIR, self).__init__()
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(shufflenet.children())[:-1])
        self.cbam = CBAM(1024)
        self.bilstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = x.view(batch_size, 1, -1)
        lstm_out, _ = self.bilstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out

# ==== Dummy Dataset for Structure ====
class DummyWeedDataset(Dataset):
    def __init__(self, folder):
        self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = 0 if 'crop' in self.images[idx].lower() else 1
        return self.transform(image), label

    def __len__(self):
        return len(self.images)

# ==== Training Loop ====
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RDODL_WRMAIR().to(device)
    dataset = DummyWeedDataset('data/sample')  # replace with actual path
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(10):  # reduce for demo
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100 * correct / len(dataset)
        print(f"Epoch [{epoch+1}/10], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train_model()

