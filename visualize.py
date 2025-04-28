import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from utils import register_cardd_dataset

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # تغییر بده به تعداد کلاس‌های دیتاست
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    return predictor

def visualize_predictions():
    register_cardd_dataset()
    predictor = load_model()

    test_images = os.listdir("./datasets/CarDD/val")
    os.makedirs("./output_images", exist_ok=True)

    for img_name in test_images:
        img_path = os.path.join("./datasets/CarDD/val", img_name)
        im = cv2.imread(img_path)
        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1], scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

        save_path = os.path.join("./output_images", img_name)
        cv2.imwrite(save_path, out)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    visualize_predictions()
