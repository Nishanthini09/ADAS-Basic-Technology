import cv2
from ultralytics import YOLO

# Load YOLOv8 model (use yolov8n for faster, yolov8m/yolov8l for better accuracy)
model = YOLO("yolov8n.pt")  # Downloaded automatically on first use

# Optional: Only keep relevant ADAS objects
ADAS_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'traffic light']

def detect_objects(frame):
    results = model(frame, imgsz=640, conf=0.3, iou=0.45)[0]
    object_info = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in ADAS_CLASSES and conf > 0.3:
            # Draw bounding box and label
            color = (0, 255, 0) if label != "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Store object for further use (e.g., tracking/speed)
            object_info.append({
                'label': label,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

    return frame, object_info
