def check_acc_and_lca(frame, detected_objects):
    h, w = frame.shape[:2]
    acc_alert = False
    lca_left = False
    lca_right = False

    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        label = obj['label']
        box_center_x = (x1 + x2) // 2

        # Check ACC – object in center and close
        if label in ['car', 'truck', 'bus'] and w * 0.4 < box_center_x < w * 0.6:
            box_height = y2 - y1
            if box_height > h * 0.3:  # Object is large = close
                acc_alert = True

        # LCA – objects on left/right sides (blind spots)
        if label in ['car', 'truck', 'bus', 'motorcycle']:
            if box_center_x < w * 0.35:
                lca_left = True
            elif box_center_x > w * 0.65:
                lca_right = True

    return acc_alert, lca_left, lca_right
