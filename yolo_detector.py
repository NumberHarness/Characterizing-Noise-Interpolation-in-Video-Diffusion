"""Shared YOLO-only detector used by experiment readout scripts."""
from __future__ import annotations

_MODEL = None
_OWL_MODEL = None
_OWL_PROCESSOR = None
OWL_MODEL_ID = "google/owlv2-large-patch14-ensemble"
OWL_MODEL_REVISION = "95e26936e865f87db1742128404b3c035d47d89d"


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        from model_config import yolo_model_source

        _MODEL = YOLO(yolo_model_source())
    return _MODEL


def detect_boxes(frame_rgb, classes, *, conf: float = 0.15, imgsz: int = 1280):
    """Return YOLO boxes as (x0, y0, x1, y1, confidence) tuples."""
    result = _model().predict(
        frame_rgb,
        classes=list(classes),
        conf=conf,
        verbose=False,
        imgsz=imgsz,
    )[0]
    return [
        tuple(float(value) for value in box.xyxy[0].tolist())
        + (float(box.conf),)
        for box in result.boxes
    ]


def detect_boxes_batch(
    frames_rgb, classes, *, conf: float = 0.15, imgsz: int = 1280,
    batch_size: int = 8,
):
    """Detect subjects in sampled frames together to reduce model-call overhead."""
    frames = list(frames_rgb)
    if not frames:
        return []
    results = _model().predict(
        frames,
        classes=list(classes),
        conf=conf,
        verbose=False,
        imgsz=imgsz,
        batch=min(batch_size, len(frames)),
    )
    return [
        [
            tuple(float(value) for value in box.xyxy[0].tolist())
            + (float(box.conf),)
            for box in result.boxes
        ]
        for result in results
    ]


def detect_owl_box(frame_rgb, classes, *, conf: float = 0.10):
    """Use public OWLv2 as the k-order probe's fallback for YOLO misses."""
    global _OWL_MODEL, _OWL_PROCESSOR
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    if _OWL_MODEL is None:
        _OWL_MODEL = Owlv2ForObjectDetection.from_pretrained(
            OWL_MODEL_ID, revision=OWL_MODEL_REVISION,
            torch_dtype=torch.float16,
        ).to("cuda").eval()
        _OWL_PROCESSOR = Owlv2Processor.from_pretrained(
            OWL_MODEL_ID, revision=OWL_MODEL_REVISION
        )

    prompts = {
        8: ["a boat"],
        17: ["a horse"],
        19: ["a cow"],
    }
    queries = next((prompts[class_id] for class_id in classes
                    if class_id in prompts), ["an object"])
    image = Image.fromarray(frame_rgb)
    inputs = _OWL_PROCESSOR(
        images=image, text=[queries], return_tensors="pt"
    ).to("cuda", torch.float16)
    with torch.no_grad():
        output = _OWL_MODEL(**inputs)
    logits = output.logits[0].float()
    boxes = output.pred_boxes[0].float()
    if logits.shape[-1] == len(queries) + 1:
        logits = logits[:, :-1]
    scores = logits.sigmoid()
    query_index, label_index = divmod(
        int(scores.argmax()), scores.shape[-1]
    )
    score = float(scores[query_index, label_index])
    if score < conf:
        return None
    cx, cy, width, height = boxes[query_index].tolist()
    image_width, image_height = image.size
    return (
        cx * image_width - width * image_width / 2,
        cy * image_height - height * image_height / 2,
        cx * image_width + width * image_width / 2,
        cy * image_height + height * image_height / 2,
        score,
    )


def detect_boxes_with_owl_fallback(
    frame_rgb, classes, *, conf: float = 0.15, imgsz: int = 1280,
    owl_conf: float = 0.10,
):
    """Return YOLO boxes, falling back to a single OWLv2 box on a YOLO miss."""
    boxes = detect_boxes(frame_rgb, classes, conf=conf, imgsz=imgsz)
    if boxes:
        return boxes
    box = detect_owl_box(frame_rgb, classes, conf=owl_conf)
    return [box] if box is not None else []
