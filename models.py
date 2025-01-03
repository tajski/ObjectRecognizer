from ultralytics import YOLO

def load_custom_model():
    return YOLO("model_custom.pt")

def load_coco_model():
    return YOLO("model_coco128.pt")
