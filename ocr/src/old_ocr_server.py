# ocr_server.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import io
import base64
from pydantic import BaseModel, Field
from typing import List, Union

# Use absolute import as discussed
from src.ocr_manager import OCRManager # Make sure this matches your ocr_manager.py location

import logging # Add logging for consistency if desired

# --- Pydantic Models for Request and Response ---
class OCRInstance(BaseModel):
    key: Union[int, str] = Field(..., description="Unique identifier for the instance")
    b64: str = Field(..., description="Base64 encoded JPEG image string")

class SimplerOCRRequest(BaseModel):
    instances: List[OCRInstance]

app = FastAPI(
    title="RapidOCR Service for OCR Challenge", # Updated title
    description="Accepts base64 encoded images and returns OCR transcriptions using RapidOCR.",
    version="1.2_rapidocr" 
)

# Global instance, initialized at module load time
try:
    print("Attempting to initialize global OCRManager instance...")
    ocr_manager_instance = OCRManager()
    print(f"Global OCRManager instance created: {'OK' if ocr_manager_instance and ocr_manager_instance.engine else 'ENGINE FAILED TO LOAD'}")
except Exception as e:
    print(f"CRITICAL: Failed to initialize global OCRManager at module load: {e}")
    import traceback
    traceback.print_exc() 
    ocr_manager_instance = None 

@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup event triggered.")
    if ocr_manager_instance is not None:
        app.state.ocr_manager = ocr_manager_instance # Make it accessible via app.state
        if app.state.ocr_manager.engine is not None:
            print("OCR Manager (RapidOCR) and its engine are initialized and ready.")
        else:
            print("ERROR: OCR Manager (RapidOCR) is initialized, but its engine failed to load. OCR will not work.")
    else:
        print("CRITICAL ERROR: Global OCRManager instance is None. OCR functionality disabled.")
        app.state.ocr_manager = None # Ensure app.state.ocr_manager exists even if None


@app.post("/ocr", response_model=None) 
async def process_ocr_challenge(
    payload: SimplerOCRRequest = Body(...) 
):
    # Access manager via app.state for consistency, though global ocr_manager_instance could also be used
    current_ocr_manager = app.state.ocr_manager 

    if not current_ocr_manager or not current_ocr_manager.engine:
        # Log this or print
        print("OCR Service not available: Manager or Engine not initialized.")
        raise HTTPException(status_code=503, detail="OCR Service not available due to initialization error.")

    predictions = []
    USE_LAYOUT_FOR_THIS_ENDPOINT = False # RapidOCR manager handles layout internally

    for instance in payload.instances:
        try:
            print(f"Processing instance with key: {instance.key}")
            # Ensure b64 string is not empty before decoding
            if not instance.b64:
                print(f"Warning: Empty base64 string for key {instance.key}.")
                predictions.append(f"Error: Empty base64 data for key {instance.key}")
                continue
            
            try:
                # Add padding if necessary, as some base64 decoders are strict
                missing_padding = len(instance.b64) % 4
                if missing_padding:
                    b64_padded = instance.b64 + '=' * (4 - missing_padding)
                else:
                    b64_padded = instance.b64
                image_bytes = base64.b64decode(b64_padded)

                if not image_bytes: 
                    print(f"Warning: Base64 decoding resulted in empty bytes for key {instance.key}.")
                    predictions.append(f"Error: Empty image data after decoding for key {instance.key}") 
                    continue
            except base64.binascii.Error as b64_error:
                print(f"Base64 decoding error for key {instance.key}: {b64_error}")
                predictions.append(f"Error: Invalid base64 string for key {instance.key}") 
                continue
            
            result_dict = current_ocr_manager.ocr(io.BytesIO(image_bytes), use_layout_analysis=USE_LAYOUT_FOR_THIS_ENDPOINT)
            
            if "error" in result_dict and result_dict["error"]: # Check if error is not None or empty
                print(f"OCR processing error for key {instance.key}: {result_dict['error']}")
                predictions.append("") # Return empty string for error as per previous logic
            else:
                transcript = result_dict.get("text", "")
                predictions.append(transcript)
                print(f"  Instance {instance.key} transcript length: {len(transcript)}")

        except Exception as e:
            print(f"Unexpected error processing instance key {instance.key}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append("") # Return empty string for error

    if len(predictions) != len(payload.instances):
        print(f"Error: Mismatch in number of predictions ({len(predictions)}) and instances ({len(payload.instances)}).")
        # Pad with error messages if lengths don't match
        while len(predictions) < len(payload.instances):
            predictions.append("Processing error - prediction/instance count mismatch") 

    return {"predictions": predictions}


@app.get("/health")
async def health_check():
    # Check app.state.ocr_manager which is set during startup
    if app.state.ocr_manager and app.state.ocr_manager.engine:
        return {"status": "healthy", "message": "RapidOCR Service is running and engine is initialized."}
    elif app.state.ocr_manager:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "RapidOCR Service is running, but OCR engine failed to initialize."})
    else:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "RapidOCR Service (OCRManager) failed to initialize at all."})

# If you run this file directly (e.g. python src/ocr_server.py), this part is used.
# However, with Docker CMD ["uvicorn", "src.ocr_server:app"...], Uvicorn handles the server run.
# import os
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5003))
#     uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")