import base64
import io
from typing import Any, List
from fastapi import FastAPI, Request
from PIL import Image

# Use a relative import for better package robustness
from cv_manager import CVManager

app = FastAPI()
manager = CVManager(model_path="best.pt")

@app.post("/cv")
async def cv(request: Request) -> dict[str, List[List[dict[str, Any]]]]:
    inputs_json = await request.json()
    image_batch = [
        Image.open(io.BytesIO(base64.b64decode(instance["b64"])))
        for instance in inputs_json.get("instances", [])
    ]
    predictions = manager.cv(image_batch) if image_batch else []
    return {"predictions": predictions}

@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}