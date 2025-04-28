import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from utils import register_cardd_dataset
import os

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Adjust to your dataset
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # path to trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    return predictor

def run_inference():
    register_cardd_dataset()
    predictor = load_model()
    test_images = os.listdir("./datasets/CarDD/val")

    for img_name in test_images:
        img_path = os.path.join("./datasets/CarDD/val", img_name)
        im = cv2.imread(img_path)
        outputs = predictor(im)

        v = detectron2.utils.visualizer.Visualizer(im[:, :, ::-1], scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = v.get_image()[:, :, ::-1]

        output_path = os.path.join("./output_images", img_name)
        cv2.imwrite(output_path, out)

if __name__ == "__main__":
    run_inference()
