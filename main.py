import cv2
from lane_detection import detect_lanes
from object_detection import detect_objects
from acc_lca import check_acc_and_lca
from lka import lane_keeping_assist
from utils import draw_label

cap = cv2.VideoCapture("project_video.mp4")  # or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    lane_output = detect_lanes(frame.copy())
    lane_frame = lane_output[0]
    left_line = lane_output[1] if len(lane_output) > 1 else None
    right_line = lane_output[2] if len(lane_output) > 2 else None

    lka_frame, deviation, lka_warning = lane_keeping_assist(lane_frame, left_line, right_line)
    object_frame, detected_objects = detect_objects(lka_frame)
    acc_alert, lca_left, lca_right = check_acc_and_lca(object_frame, detected_objects)

    if acc_alert:
        draw_label(object_frame, "ACC Alert: Slow down!", (10, 40), color=(0, 0, 255))
    if lca_left:
        draw_label(object_frame, "LCA Left: Vehicle in blind spot!", (10, 70), color=(0, 255, 255))
    if lca_right:
        draw_label(object_frame, "LCA Right: Vehicle in blind spot!", (10, 100), color=(0, 255, 255))
    if lka_warning:
        draw_label(object_frame, "LKA Warning: Lane drifting!", (10, 130), color=(0, 100, 255))

    cv2.imshow("ADAS - Full System View", object_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()