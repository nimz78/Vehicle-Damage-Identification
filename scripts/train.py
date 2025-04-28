# scripts/train.py

import torch
from torch.utils.data import DataLoader
import os
from utils import CarDamageDataset, get_model

def collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# لود دیتاست
train_dataset = CarDamageDataset(
    images_dir="./datasets/CarDD/train",
    annotations_path="./datasets/CarDD/annotations/train_annotations.json"
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# تعریف مدل
model = get_model(num_classes=3)
model.to(device)

# آپتیمایزر
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# آموزش
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss}")

# ذخیره مدل
os.makedirs("./output/models", exist_ok=True)
torch.save(model.state_dict(), "./output/models/model_final.pth")
