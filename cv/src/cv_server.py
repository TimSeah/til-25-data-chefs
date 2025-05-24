"""Runs the CV server."""

import base64
from typing import Any, List, Dict # List and Dict might need to be imported from typing for older Pythons but are built-in for type hints in newer ones

from fastapi import FastAPI, Request

# Changed from: from .cv_manager import CVManager
from cv_manager import CVManager  # <<< CHANGE THIS LINE


app = FastAPI()
# Assuming best.pt is also in /app
manager = CVManager(model_path="best.pt") # This should now work if best.pt is in /app


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    # ... rest of your code ...
    inputs_json = await request.json()
    predictions = []
    for instance in inputs_json["instances"]:
        image_bytes = base64.b64decode(instance["b64"])
        detections = manager.cv(image_bytes)
        predictions.append(detections)
    return {"predictions": predictions}

@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}

# If you need to run this file directly for testing (outside Docker/Uvicorn)
# you might add this, but Uvicorn handles running the app object in Docker.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5002)