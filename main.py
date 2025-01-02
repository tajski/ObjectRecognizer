from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="data_conf.yaml", epochs=50, optimizer='SGD', project='YOLOv8-Experiments', name='run2', exist_ok=True)
metrics = model.val()
print(metrics)

model_default = YOLO("yolov8n.pt")

results_default = model_default.train(data="coco128.yaml", epochs=50, lr0=0.01, project='YOLOv8-Experiments', name='run_default', exist_ok=True)
metrics_default = model_default.val()
print(metrics_default)