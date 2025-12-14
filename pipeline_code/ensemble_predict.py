import os
import math
import json
import torch
from ultralytics import YOLO
from PIL import Image

# Load models
original_model = YOLO("trained_model/best_original.pt")
negative_model = YOLO("trained_model/best_negative.pt")

METER_PER_PIXEL = 0.12   # zoom 20 approx

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0


def ensemble_predict(image_path):

    orig_pred = original_model(image_path)[0]
    neg_pred = negative_model(image_path)[0]

    orig_boxes = orig_pred.boxes
    neg_boxes  = neg_pred.boxes

    # ------------------------
    # 1. No original detection
    # ------------------------
    if len(orig_boxes) == 0:
        return False, 0.0, [], 0.0, "Original model detected nothing"

    # ------------------------
    # 2. Take strongest box
    # ------------------------
    best_idx = torch.argmax(orig_boxes.conf)
    best_box = orig_boxes[best_idx]

    orig_conf = float(best_box.conf)
    orig_xyxy = best_box.xyxy[0].tolist()

    # ------------------------
    # 3. PV area
    # ------------------------
    x1, y1, x2, y2 = orig_xyxy
    area_px = (x2 - x1) * (y2 - y1)
    area_sqm = area_px * (METER_PER_PIXEL ** 2)

    # ------------------------
    # 4. Negative suppression
    # ------------------------
    for nb in neg_boxes:
        neg_conf = float(nb.conf[0])
        neg_xyxy = nb.xyxy[0].tolist()

        if neg_conf > orig_conf and iou(orig_xyxy, neg_xyxy) > 0.3:
            return False, orig_conf, [], 0.0, "Negative model suppressed detection"

    # ------------------------
    # 5. Accept detection
    # ------------------------
    return True, orig_conf, [orig_xyxy], area_sqm, "Solar detected"
