import cv2

def draw_label(frame, text, position, color=(0, 255, 0), font_scale=0.6, thickness=2):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA
    )
