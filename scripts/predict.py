# scripts/predict.py

import torch
import os
import cv2
from utils import CarDamageDataset, get_model
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

val_dataset = CarDamageDataset(
    images_dir="./datasets/CarDD/val",
    annotations_path="./datasets/CarDD/annotations/val_annotations.json"
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_classes=3)
model.load_state_dict(torch.load("./output/models/model_final.pth", map_location=device))
model.to(device)
model.eval()

results = []

for images, targets in val_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)

    for output in outputs:
        boxes = output['boxes'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()

        results.append({
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        })

print("Prediction Done.")
