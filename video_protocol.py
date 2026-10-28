"""Shared object-selection rules for video measurements."""
from __future__ import annotations

CLASS_IDS = {
    "cow_p": [19],
    "dog_p": [16, 19, 20],
    "horse_f": [17],
    "boat_l": [8],
}


def main_box(boxes, scene: str):
    """Select the tracked subject using the paper's per-scene rule."""
    if not boxes:
        return None
    if scene == "boat_l":
        signed = [box for box in boxes if 150 <= box[2] - box[0] <= 260]
        return max(signed, key=lambda box: box[4]) if signed else None
    if scene == "dog_p":
        trusted = [box for box in boxes
                   if box[2] - box[0] >= 90 and box[4] >= 0.25]
        return max(trusted,
                   key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) \
            if trusted else None
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))


def iou_box(a, b) -> float:
    """Intersection-over-union for xyxy boxes."""
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    union = ((ax1 - ax0) * (ay1 - ay0)
             + (bx1 - bx0) * (by1 - by0) - intersection)
    return intersection / union if union > 0 else 0.0
