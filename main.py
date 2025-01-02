from ultralytics import YOLO
from multiprocessing import Process, freeze_support, set_start_method

def foo():
    print('hello')

if __name__ == '__main__':
    freeze_support()
    set_start_method('spawn')
    p = Process(target=foo)
    p.start()

model = YOLO("yolov8n.yaml")

results = model.train(data="data_conf.yaml", epochs=30, imgsz=640, batch=16, augment=True, freeze=5)
