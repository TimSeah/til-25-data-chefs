# ocr_server.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import io
import base64
from pydantic import BaseModel, Field
from typing import List, Union # Added Union

# Assuming ocr_manager is in the same directory or properly in PYTHONPATH
from .ocr_manager import OCRManager 

# --- Pydantic Models for Request and Response ---
class OCRInstance(BaseModel):
    key: Union[int, str] = Field(..., description="Unique identifier for the instance") # Made key more flexible
    b64: str = Field(..., description="Base64 encoded JPEG image string")
    # Pydantic will ignore other fields like "document", "contents" from test_ocr.py by default.

class OCRRequest(BaseModel):
    instances: List[OCRInstance]
    # test_ocr.py does not send use_layout, so this field will not be populated from the request.
    # If you need to control use_layout via this request, test_ocr.py would need to send it.
    # For now, the ocr_manager_instance.ocr call will use its default for use_layout or what's passed.
    # The current call in ocr_server.py is:
    # result_dict = ocr_manager_instance.ocr(io.BytesIO(image_bytes), use_layout_analysis=payload.use_layout)
    # Since payload.use_layout isn't in the request from test_ocr.py, Pydantic might not create it,
    # or it might use a default if OCRRequest had one.
    # Let's ensure use_layout is explicitly passed or handled.
    # The `test_ocr.py` does NOT send `use_layout`.
    # The `OCRRequest` should not expect it if the client doesn't send it,
    # unless it has a default. Let's remove it from OCRRequest if test_ocr.py doesn't send it.
    # OR, the server can decide the use_layout strategy.

# Let's simplify OCRRequest based on what test_ocr.py sends:
class SimplerOCRRequest(BaseModel):
    instances: List[OCRInstance]


app = FastAPI(
    title="PaddleOCR Service for OCR Challenge",
    description="Accepts base64 encoded images and returns OCR transcriptions.",
    version="1.2_standard_model" 
)

try:
    ocr_manager_instance = OCRManager()
    print("OCR Manager instance created successfully for FastAPI app.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize OCRManager: {e}")
    import traceback
    traceback.print_exc() 
    ocr_manager_instance = None 

@app.post("/ocr", response_model=None) 
async def process_ocr_challenge(
    payload: SimplerOCRRequest = Body(...) # Using the simpler request model
):
    if not ocr_manager_instance:
        raise HTTPException(status_code=503, detail="OCR Service not available due to initialization error.")

    predictions = []
    # The server decides the layout strategy, or it's fixed in OCRManager.
    # Let's assume a fixed strategy for now, e.g., use_layout_analysis=False (or True).
    # This should ideally be configurable if needed.
    USE_LAYOUT_FOR_THIS_ENDPOINT = False # Example: Server-defined strategy

    for instance in payload.instances:
        try:
            print(f"Processing instance with key: {instance.key}")
            try:
                image_bytes = base64.b64decode(instance.b64)
                if not image_bytes: 
                    print(f"Warning: Base64 decoding resulted in empty bytes for key {instance.key}.")
                    predictions.append(f"Error: Empty image data for key {instance.key}") 
                    continue
            except base64.binascii.Error as b64_error:
                print(f"Base64 decoding error for key {instance.key}: {b64_error}")
                predictions.append(f"Error: Invalid base64 string for key {instance.key}") 
                continue
            
            result_dict = ocr_manager_instance.ocr(io.BytesIO(image_bytes), use_layout_analysis=USE_LAYOUT_FOR_THIS_ENDPOINT)
            
            if "error" in result_dict:
                print(f"OCR processing error for key {instance.key}: {result_dict['error']}")
                predictions.append("") 
            else:
                transcript = result_dict.get("text", "")
                predictions.append(transcript)
                print(f"  Instance {instance.key} transcript length: {len(transcript)}")

        except Exception as e:
            print(f"Unexpected error processing instance key {instance.key}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append("") 

    if len(predictions) != len(payload.instances):
        print(f"Error: Mismatch in number of predictions ({len(predictions)}) and instances ({len(payload.instances)}).")
        while len(predictions) < len(payload.instances):
            predictions.append("Processing error - count mismatch") 

    return {"predictions": predictions}


@app.get("/health")
async def health_check():
    if ocr_manager_instance:
        return {"status": "healthy", "message": "OCR Service is running."}
    else:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "OCR Service initialization failed."})