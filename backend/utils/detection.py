from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

FURNITURE_CLASSES = {
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    67: "cell phone",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    58: "potted plant",
}

# YOLOv8: object detection — auto-downloads yolov8n.pt on first use
_yolo_model = None


def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLOv8n model...")
        _yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded.")
    return _yolo_model


def detect_objects(pil_image: Image.Image) -> list[dict]:
    model = get_yolo_model()
    results = model(pil_image, conf=0.35, verbose=False)

    relevant_categories = {
        "chair", "couch", "sofa", "bed", "dining table", "table",
        "potted plant", "plant", "tv", "television", "refrigerator",
        "clock", "vase", "book", "lamp", "laptop", "monitor",
        "bench", "desk", "cabinet"
    }

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model.names.get(cls_id, "unknown")
            if class_name.lower() not in relevant_categories:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": class_name,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

    logger.info(f"YOLOv8 detected {len(detections)} furniture items.")
    return detections
