import cv2
import numpy as np
from typing import Any, List, Dict
from ultralytics import YOLO


class CVManager:
    def __init__(self, model_path: str = "best.pt"):
        """
        Initialize the YOLOv8 model for inference.

        Args:
            model_path: Path to your trained YOLOv8 weights file (.pt).
                        Update this to your fine-tuned checkpoint for the 18-class dataset.
        """
        # Load the YOLOv8 model (in eval mode by default).
        self.model = YOLO(model_path)

    def cv(self, image: bytes) -> List[Dict[str, Any]]:
        """
        Perform object detection on a JPEG image.

        Args:
            image: Raw JPEG bytes.

        Returns:
            A list of detections, each a dict:
              {
                "bbox": [x, y, w, h],
                "category_id": <int 0–17>
              }
        """
        # 1) Decode bytes to a BGR image (height, width, 3)
        arr = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # 2) Run YOLO inference (adjust conf and iou thresholds as needed)
        results = self.model(img, conf=0.25, iou=0.45)[0]

        # 3) Parse predictions into required format
        preds: List[Dict[str, Any]] = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            preds.append({
                "bbox": [x, y, w, h],
                "category_id": int(cls)
            })

        return preds
