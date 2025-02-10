from ultralytics import YOLO
import cv2
import torch

# Load trained YOLOv8 model
model = YOLO("xray_yolov8.pt")

# Test image path
image_path = r""

# Load image using OpenCV
img = cv2.imread(image_path)

# Ensure the model runs on CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Run YOLO inference on the image
results = model(image_path, conf=0.2) 

# Check if there are any detections
for result in results:
    if len(result.boxes) == 0:
        print("⚠️ No objects detected! Try lowering the confidence threshold or retraining.")
    else:
        print(f"✅ Detected {len(result.boxes)} objects!")

# Show results (this now works correctly)
results[0].show()

# Optional: Draw boxes manually if show() doesn't work
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("YOLOv8 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
