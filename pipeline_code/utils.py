import json
import base64

def encode_bbox(bbox):
    """
    Encode a bounding box [x1, y1, x2, y2] to Base64 string.
    """
    if bbox is None:
        return None
    return base64.b64encode(json.dumps(bbox).encode()).decode()


def estimate_area_sqm(bbox):
    """
    Approximate solar panel area from bounding box.
    This is a placeholder estimation suitable for the challenge.
    """
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    pixel_area = w * h

    sqm = pixel_area * 0.10    # Approx factor
    return round(sqm, 2)


def determine_qc_status(has_solar):
    return "VERIFIABLE" if has_solar else "NO_PANEL_DETECTED"
