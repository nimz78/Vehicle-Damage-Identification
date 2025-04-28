# scripts/visualize.py

import os
import cv2
import torch
import numpy as np
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

os.makedirs("./output/images", exist_ok=True)

for images, targets in val_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)

    img = images[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)

    masks = outputs[0]['masks']
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    for i in range(len(masks)):
        if scores[i] > 0.5:
            box = boxes[i].detach().cpu().numpy().astype(int)
            label = labels[i].item()

            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            img = cv2.putText(img, str(label), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    filename = f"./output/images/val_{targets[0]['image_id'].item()}.jpg"
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img_bgr)
