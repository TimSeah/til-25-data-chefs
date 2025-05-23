# ocr_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import io
from .ocr_manager import OCRManager 

app = FastAPI(
    title="PaddleOCR Service",
    description="Provides OCR capabilities using PaddleOCR, including optional layout analysis.",
    version="1.0._YOUR_FINETUNED_MODEL_VERSION_" 
)

try:
    ocr_manager_instance = OCRManager()
    print("OCR Manager instance created successfully for FastAPI app.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize OCRManager: {e}")
    ocr_manager_instance = None 


@app.post("/ocr/")
async def process_ocr(
    image: UploadFile = File(..., description="Image file to perform OCR on."),
    use_layout: bool = Form(False, description="Set to true to use layout analysis for potentially better structure extraction (e.g., columns).")
):
    if not ocr_manager_instance:
        raise HTTPException(status_code=503, detail="OCR Service not available due to initialization error.")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await image.read()
        
        print(f"Received image: {image.filename}, content type: {image.content_type}, size: {len(image_bytes)} bytes")
        print(f"Performing OCR with use_layout={use_layout}")

        # Updated method call here
        result = ocr_manager_instance.ocr(io.BytesIO(image_bytes), use_layout_analysis=use_layout)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        print(f"OCR Result text length: {len(result.get('text', ''))}")
        if 'layout_boxes' in result and result['layout_boxes']:
            print(f"Layout analysis returned {len(result['layout_boxes'])} regions.")

        return JSONResponse(content=result)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during OCR: {str(e)}")

@app.get("/health")
async def health_check():
    if ocr_manager_instance:
        return {"status": "healthy", "message": "OCR Service is running."}
    else:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "OCR Service initialization failed."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)