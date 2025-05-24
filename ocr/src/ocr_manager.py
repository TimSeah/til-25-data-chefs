# ocr_manager.py
import os
import numpy as np
from PIL import Image
import easyocr
import io # Make sure io is imported

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager with EasyOCR...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=True) 
            print("EasyOCR Reader initialized successfully.")
        except Exception as e:
            print(f"CRITICAL: Failed to initialize EasyOCR Reader: {e}")
            import traceback
            traceback.print_exc()
            self.reader = None

    def _preprocess_image(self, image_data_wrapper): # Renamed for clarity
        """
        Converts image_data_wrapper (which is an io.BytesIO object) to a PIL Image object.
        """
        try:
            # image_data_wrapper is an io.BytesIO object
            image = Image.open(image_data_wrapper).convert("RGB")
            return image
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None

    def ocr(self, image_data_wrapper, use_layout_analysis=False): # Renamed for clarity
        """
        Performs OCR using EasyOCR.
        image_data_wrapper is an io.BytesIO object from ocr_server.py
        """
        if not self.reader:
            print("ERROR: EasyOCR Reader is not available.")
            return {"error": "EasyOCR not initialized", "text": ""}

        # Preprocessing is optional if EasyOCR can handle the raw bytes directly well
        # For now, let's try passing raw bytes first, as it's simpler.
        # pil_image = self._preprocess_image(image_data_wrapper)
        # if pil_image is None:
        #     return {"error": "Failed to preprocess image", "text": ""}
        # image_for_easyocr = np.array(pil_image) # Option: Convert PIL to NumPy array

        # Get raw bytes from the io.BytesIO wrapper
        raw_image_bytes = image_data_wrapper.getvalue() # Or .read()

        if not raw_image_bytes:
            return {"error": "Empty image data received", "text": ""}

        extracted_text = ""
        try:
            # Pass the raw_image_bytes directly
            result = self.reader.readtext(raw_image_bytes, detail=0, paragraph=True)
            extracted_text = "\n".join(result)
            
            print(f"EasyOCR extracted text (first 100 chars): {extracted_text[:100]}")

        except Exception as e:
            print(f"Error during EasyOCR processing: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"OCR processing error: {str(e)}", "text": ""}
        
        return {"text": extracted_text.strip()}
