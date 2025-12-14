import os
import json
import math
import requests
from io import BytesIO
from PIL import Image
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# Load YOLO Models (Original + Negative)
# -----------------------------------------------------
original_model = YOLO("trained_model/best_original.pt")
negative_model = YOLO("trained_model/best_negative.pt")


# -----------------------------------------------------
# IoU Calculation
# -----------------------------------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# -----------------------------------------------------
# Fetch Google Satellite Image
# -----------------------------------------------------
def fetch_satellite_image(lat, lon, zoom=20, size=640):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}x{size}"
        f"&maptype=satellite&key={api_key}"
    )

    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    path = "temp_satellite.png"
    img.save(path)
    return path


# -----------------------------------------------------
# Compute PV Area in Sq Meters
# -----------------------------------------------------
def compute_area_sqm(x1, y1, x2, y2, zoom=20):
    pixel_area = (x2 - x1) * (y2 - y1)

    METERS_PER_PIXEL = 0.12  # Google zoom-20 approximation
    sqm = pixel_area * (METERS_PER_PIXEL ** 2)
    return sqm


# -----------------------------------------------------
# Ensemble Prediction Logic
# -----------------------------------------------------
def ensemble_predict(image_path):
    orig_pred = original_model(image_path)[0]
    neg_pred = negative_model(image_path)[0]

    orig_boxes = orig_pred.boxes
    neg_boxes = neg_pred.boxes

    # No solar detected
    if len(orig_boxes) == 0:
        return False, 0.0, [], 0.0, "Original model found nothing"

    # Pick best original detection
    best_idx = torch.argmax(orig_boxes.conf)
    best_box = orig_boxes[best_idx]
    orig_conf = float(best_box.conf)
    orig_xyxy = best_box.xyxy[0].tolist()

    # Compute panel area
    x1, y1, x2, y2 = orig_xyxy
    pv_area = compute_area_sqm(x1, y1, x2, y2)

    # If strong detection -> accept
    if orig_conf > 0.55:
        return True, orig_conf, [orig_xyxy], pv_area, "High-confidence solar detection"

    # Check negative model suppression
    for nb in neg_boxes:
        neg_conf = float(nb.conf[0])
        neg_xyxy = nb.xyxy[0].tolist()

        if neg_conf > 0.40 and iou(orig_xyxy, neg_xyxy) > 0.30:
            return False, orig_conf, [], 0.0, "Suppressed by negative model"

    # Default accept
    return True, orig_conf, [orig_xyxy], pv_area, "Accepted after ensemble check"


# -----------------------------------------------------
# Build JSON response
# -----------------------------------------------------
def build_output_json(sample_id, lat, lon, has_solar, confidence, bboxes, area_sqm):
    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": has_solar,
        "confidence": round(confidence, 3),
        "pv_area_sqm_est": round(area_sqm, 2),
        "buffer_radius_sqft": 1200,
        "qc_status": "VERIFIABLE",
        "bbox_or_mask": bboxes,
        "image_metadata": {
            "source": "Google Static Maps",
            "capture_date": "N/A"
        }
    }
