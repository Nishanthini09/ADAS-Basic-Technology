import cv2
from object_detection import detect_objects
from utils import draw_label

cap = cv2.VideoCapture("project_video.mp4")
frame_width = 960
frame_height = 540

# Configurable thresholds
DANGER_HEIGHT = 70  # object box height in pixels
MIN_CONFIDENCE = 0.5
LEFT_ZONE = frame_width // 3
RIGHT_ZONE = 2 * frame_width // 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    obj_frame, objects = detect_objects(frame.copy())

    acc_alert = False
    lca_left = False
    lca_right = False

    for obj in objects:
        label = obj["label"]
        conf = obj["confidence"]
        x1, y1, x2, y2 = obj["bbox"]
        center_x = (x1 + x2) // 2
        height = y2 - y1

        if conf < MIN_CONFIDENCE:
            continue

        # ACC: danger in center
        if label in ["car", "bus", "truck", "motorbike"] and LEFT_ZONE < center_x < RIGHT_ZONE:
            if height >= DANGER_HEIGHT:
                acc_alert = True

        # LCA: detect left/right zones
        if label in ["car", "bus", "truck", "motorbike"]:
            if center_x < LEFT_ZONE:
                lca_left = True
            elif center_x > RIGHT_ZONE:
                lca_right = True

        # Draw bounding box
        cv2.rectangle(obj_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label(obj_frame, f"{label} ({conf:.2f})", (x1, y1 - 10))

    # Optional: draw debug zones
    cv2.line(obj_frame, (LEFT_ZONE, 0), (LEFT_ZONE, frame_height), (255, 255, 0), 1)
    cv2.line(obj_frame, (RIGHT_ZONE, 0), (RIGHT_ZONE, frame_height), (255, 255, 0), 1)

    # Display alerts
    if acc_alert:
        draw_label(obj_frame, "⚠️ ACC: Vehicle Ahead", (20, 40), color=(0, 0, 255), font_scale=0.8)
    if lca_left:
        draw_label(obj_frame, "⬅️ LCA: Left Lane Blocked", (20, 70), color=(0, 200, 255), font_scale=0.7)
    if lca_right:
        draw_label(obj_frame, "➡️ LCA: Right Lane Blocked", (20, 100), color=(0, 200, 255), font_scale=0.7)

    cv2.imshow("ADAS - ACC + LCA", obj_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
