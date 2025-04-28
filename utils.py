from detectron2.data.datasets import register_coco_instances

def register_cardd_dataset():
    register_coco_instances(
        "cardd_train",
        {},
        "./datasets/CarDD/annotations/train.json",
        "./datasets/CarDD/train"
    )
    register_coco_instances(
        "cardd_val",
        {},
        "./datasets/CarDD/annotations/val.json",
        "./datasets/CarDD/val"
    )
