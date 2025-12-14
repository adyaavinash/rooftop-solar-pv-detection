import os
import sys
import json
import math
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import torch
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Load models
# -----------------------------
original_model = YOLO("trained_model/best_original.pt")
negative_model = YOLO("trained_model/best_negative.pt")

# -----------------------------
# IoU function
# -----------------------------
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

# -----------------------------
# Fetch Google Satellite Image
# -----------------------------
def fetch_satellite_image(lat, lon, zoom=20, size=640):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}x{size}"
        f"&maptype=satellite&key={api_key}"
    )

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    temp_path = "temp_satellite.png"
    img.save(temp_path)

    return temp_path

# -----------------------------
# Compute PV Area in Sq Meters
# -----------------------------
def compute_area_sqm(x1, y1, x2, y2, zoom=20):
    width = x2 - x1
    height = y2 - y1
    pixel_area = width * height

    # Google Maps approx meters per pixel at zoom 20
    METERS_PER_PIXEL = 0.12
    sqm = pixel_area * (METERS_PER_PIXEL ** 2)
    return sqm

# -----------------------------
# Ensemble prediction
# -----------------------------
def ensemble_predict(image_path):
    orig_pred = original_model(image_path)[0]
    neg_pred = negative_model(image_path)[0]

    orig_boxes = orig_pred.boxes
    neg_boxes = neg_pred.boxes

    # If no detections → No solar
    if len(orig_boxes) == 0:
        return False, 0.0, [], 0.0, "Original model found nothing"

    # Best original detection
    best_idx = torch.argmax(orig_boxes.conf)
    best_box = orig_boxes[best_idx]
    orig_conf = float(best_box.conf)
    orig_xyxy = best_box.xyxy[0].tolist()

    # Compute area
    x1, y1, x2, y2 = orig_xyxy
    pv_area = compute_area_sqm(x1, y1, x2, y2)

    # If original confidence is high → accept directly
    if orig_conf > 0.55:
        return True, orig_conf, [orig_xyxy], pv_area, "High-confidence solar detection"

    # Suppression by the negative model
    for nb in neg_boxes:
        neg_conf = float(nb.conf[0])
        neg_xyxy = nb.xyxy[0].tolist()

        if neg_conf > 0.40 and iou(orig_xyxy, neg_xyxy) > 0.30:
            return False, orig_conf, [], 0.0, "Suppressed by negative model"

    # Accept if negative model did not contradict
    return True, orig_conf, [orig_xyxy], pv_area, "Accepted after ensemble check"

# -----------------------------
# Build JSON Output
# -----------------------------
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

# -----------------------------
# Main runner
# -----------------------------
def run(lat, lon, output_file):
    img = fetch_satellite_image(lat, lon)
    has_solar, conf, boxes, pv_area, reason = ensemble_predict(img)

    result = build_output_json(
        sample_id=1,
        lat=lat,
        lon=lon,
        has_solar=has_solar,
        confidence=conf,
        bboxes=boxes,
        area_sqm=pv_area
    )

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print("\n=== PREDICTION COMPLETE ===")
    print("Reason:", reason)
    print("Saved →", output_file)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    output_file = sys.argv[3]
    run(lat, lon, output_file)
