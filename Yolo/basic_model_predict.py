from ultralytics import YOLO

# Load the detection model
model = YOLO('yolov8n.pt')
model.predict(source="foot1.mp4")