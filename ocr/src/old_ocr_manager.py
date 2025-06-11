import io
import time
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR # Import RapidOCR
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class OCRManager:
    def __init__(self):
        """
        Initializes the OCRManager with RapidOCR.
        Models are typically downloaded automatically by RapidOCR on first use.
        """
        self.engine_name = "RapidOCR (ONNXRuntime)"
        logger.info(f"Initializing OCR engine: {self.engine_name}")
        try:
            # RapidOCR() will initialize the models.
            # It typically auto-detects available hardware (CPU/GPU for ONNXRuntime if built with CUDA support).
            self.engine = RapidOCR()
            logger.info(f"{self.engine_name} initialized successfully.")
            # Perform a dummy inference to ensure models are downloaded/loaded if that's desired at init
            # For example: self.engine(np.zeros((100,100,3), dtype=np.uint8)) 
            # logger.info("Dummy inference successful (models likely loaded/downloaded).")
        except Exception as e:
            logger.error(f"Error initializing {self.engine_name}: {e}", exc_info=True)
            self.engine = None

    def _preprocess_image(self, image_data: io.BytesIO) -> np.ndarray | None:
        """
        Loads image from BytesIO, converts to RGB NumPy array.
        RapidOCR generally expects a BGR NumPy array or an image path.
        Pillow opens images in RGB by default.
        """
        try:
            img = Image.open(image_data)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_np_rgb = np.array(img)
            # Convert RGB to BGR as RapidOCR (like many OpenCV-based models) might prefer BGR
            img_np_bgr = img_np_rgb[:, :, ::-1] 
            logger.debug(f"Image preprocessed successfully. Shape: {img_np_bgr.shape}")
            return img_np_bgr
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}", exc_info=True)
            return None

    def ocr(self, image_data: io.BytesIO, use_layout_analysis: bool = False) -> dict:
        """
        Performs OCR on the given image data.

        Args:
            image_data: A BytesIO object containing the image.
            use_layout_analysis: This flag is noted but RapidOCR's layout handling
                                 is intrinsic to its detection/recognition process.
                                 We will use RapidOCR's default behavior.

        Returns:
            A dictionary containing the recognized text under the "text" key,
            or an "error" key if an error occurred.
        """
        if self.engine is None:
            logger.error("OCR engine not initialized. Cannot perform OCR.")
            return {"text": "", "error": "OCR engine not initialized"}

        start_time = time.time()
        
        img_np = self._preprocess_image(image_data)
        if img_np is None:
            return {"text": "", "error": "Image preprocessing failed"}

        logger.debug(f"Sending image data (type: {type(img_np)}, shape: {img_np.shape}) to {self.engine_name}...")
        
        full_text = ""
        try:
            # RapidOCR's __call__ method performs detection and recognition.
            # Result is typically a list of tuples: (bounding_box, text, confidence_score)
            # or None if no text is found.
            ocr_results, _ = self.engine(img_np) # The second return is usually timing info

            if ocr_results is not None:
                # Extract and join all recognized text segments
                text_segments = [res[1] for res in ocr_results if res and len(res) > 1]
                full_text = " ".join(text_segments).strip()
                logger.debug(f"Raw joined text from {self.engine_name} (before strip, length {len(full_text)}): '{full_text[:200]}...'")
            else:
                logger.info("No text detected by RapidOCR.")
                full_text = ""

        except Exception as e:
            logger.error(f"Error during {self.engine_name} inference: {e}", exc_info=True)
            return {"text": "", "error": f"{self.engine_name} inference error: {str(e)}"}
        finally:
            end_time = time.time()
            logger.info(f"OCR processing took {end_time - start_time:.4f} seconds. Recognized text length: {len(full_text)}")
            
        return {"text": full_text, "error": None}
