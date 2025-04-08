import cv2
import numpy as np

def lane_keeping_assist(frame, left_line, right_line):
    h, w = frame.shape[:2]
    deviation = 0
    warning = False

    # Check if both lanes exist
    if left_line is not None and right_line is not None:
        # Get lane centers
        left_x = int((left_line[0][0] + left_line[1][0]) / 2)
        right_x = int((right_line[0][0] + right_line[1][0]) / 2)
        lane_center = int((left_x + right_x) / 2)

        # Car assumed center at bottom
        car_center = w // 2
        deviation = car_center - lane_center

        # Visualize
        cv2.line(frame, (car_center, h), (lane_center, h - 100), (0, 255, 255), 2)
        cv2.putText(frame, f'Deviation: {deviation}px', (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Trigger warning if deviation too large
        if abs(deviation) > 50:
            warning = True
            cv2.putText(frame, 'LKA Warning: Lane Drifting!', (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, deviation, warning
