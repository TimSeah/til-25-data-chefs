from typing import Any, List, Dict
from ultralytics import YOLO
from PIL import Image # <<< ADD THIS IMPORT

class CVManager:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)
        print(f"YOLO model loaded successfully from {model_path}")

    def cv(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]: # <<< CORRECTED TYPE HINT
        if not self.model or not images:
            return [[] for _ in images]

        all_predictions = []
        try:
            results = self.model(
                images,
                device='cuda',
                half=True,
                verbose=False,
                imgsz=640, # CRITICAL: Must match training size
                conf=0.25,
                iou=0.45
            )
            for result in results:
                preds_for_one_image: List[Dict[str, Any]] = []
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = box.tolist()
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    preds_for_one_image.append({
                        "bbox": [x, y, w, h],
                        "category_id": int(cls)
                    })
                all_predictions.append(preds_for_one_image)
        except Exception as e:
            print(f"An error occurred during CV batch processing: {e}")
            return [[] for _ in images]
        return all_predictions