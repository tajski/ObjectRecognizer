from ultralytics import YOLO


model = YOLO("yolov8n.pt")
results = model.train(data="data_conf.yaml", epochs=50, optimizer='SGD', project='YOLOv8-Experiments', name='custom', exist_ok=True)
model.save("model_custom.pt")

metrics = model.val()
print(metrics)


model_coco = YOLO("yolov8n.pt")
results_coco = model_coco.train(data="coco128.yaml", epochs=50, lr0=0.01, project='YOLOv8-Experiments', name='yolo_default', exist_ok=True)
model_coco.save("model_coco128.pt")

metrics_coco = model_coco.val()
print(metrics_coco)
