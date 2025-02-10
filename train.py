from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" for fast training

# Train on X-ray dataset
model.train(data=r"data.yaml", epochs=50, batch=8, imgsz=640)

# Save the trained model
model.save("xray_yolov8.pt")
