# ocr_manager.py
import os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, PPStructure
# import yaml # PyYAML - Not strictly needed if we are not loading YAML configs here

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager...")

        # --- Model Path Configuration from Environment Variables ---
        self.models_base_dir = os.getenv('MODELS_BASE_DIR', '/opt/paddleocr_models')
        
        # Detection Model - This path is correct as per your Dockerfile for the standard model
        self.det_model_dir = os.path.join(self.models_base_dir, os.getenv('DET_MODEL_SUBDIR', 'det/en/en_PP-OCRv3_det_infer'))
        
        # Recognition Model - We will NOT use a specific rec_model_dir for standard pre-trained model.
        # PaddleOCR will use its default for lang='en'.
        # self.rec_model_dir = os.path.join(self.models_base_dir, os.getenv('REC_MODEL_SUBDIR', 'rec/en/en_PP-OCRv4_rec_infer')) 
        
        # Classification Model - This path is correct as per your Dockerfile for the standard model
        self.cls_model_dir = os.path.join(self.models_base_dir, os.getenv('CLS_MODEL_SUBDIR', 'cls/en/ch_ppocr_mobile_v2.0_cls_infer'))
        
        # Layout Model - This path is correct as per your Dockerfile for the standard model
        self.layout_model_dir = os.path.join(self.models_base_dir, os.getenv('LAYOUT_MODEL_SUBDIR', 'layout/en/picodet_lcnet_x1_0_fgd_layout_infer'))

        # Character Dictionary - We will NOT use a custom char dict for standard pre-trained model.
        # self.rec_char_dict_path = os.path.join(self.models_base_dir, 'dicts/custom_char_dict.txt')

        print(f"  Models Base Dir: {self.models_base_dir}")
        print(f"  Det Model Dir: {self.det_model_dir}")
        # print(f"  Rec Model Dir: Using default for lang='en'") # Updated print
        print(f"  Cls Model Dir: {self.cls_model_dir}")
        print(f"  Layout Model Dir: {self.layout_model_dir}")
        # print(f"  Rec Char Dict Path: Using default for lang='en'") # Updated print

        if not os.path.exists(self.det_model_dir):
            print(f"WARNING: Detection model directory not found: {self.det_model_dir}. Ensure it's correctly placed/downloaded by Dockerfile.")
        # No need to check for self.rec_model_dir or self.rec_char_dict_path if using defaults
        if not os.path.exists(self.cls_model_dir):
            print(f"WARNING: Classification model directory not found: {self.cls_model_dir}. Ensure it's correctly placed/downloaded by Dockerfile.")
        if not os.path.exists(self.layout_model_dir):
            print(f"WARNING: Layout model directory not found: {self.layout_model_dir}. Ensure it's correctly placed/downloaded by Dockerfile.")


        print("Initializing PaddleOCR engine with standard English pre-trained models...")
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='en',  # This will use default English models for detection and recognition
            det_model_dir=self.det_model_dir, # Use the standard detection model from Dockerfile
            # rec_model_dir=self.rec_model_dir, # DO NOT specify for default recognition model
            # rec_char_dict_path=self.rec_char_dict_path, # DO NOT specify for default char dict
            cls_model_dir=self.cls_model_dir, # Use the standard classification model from Dockerfile
            use_gpu=True, # Assuming GPU is available as per Dockerfile
            show_log=True # Good for debugging
        )
        print("PaddleOCR engine initialized.")

        print("Initializing PPStructure engine...")
        self.structure_engine = PPStructure(
            # layout_model_dir is specified, which is fine.
            # PPStructure can also use the ocr_engine passed to it or its own OCR instance.
            # For simplicity, let's assume it will use its default OCR or the one we configure if needed.
            # If you want PPStructure to use the *exact same* OCR engine instance:
            # ocr_engine_for_ppstructure = self.ocr_engine # or a new instance configured similarly
            # table_model_dir, layout_model_dir, etc.
            layout_model_dir=self.layout_model_dir,
            show_log=True,
            use_gpu=True, # Assuming GPU
            # layout_dict_path=None # Default is fine
            # We can also pass the configured ocr_engine to PPStructure if we want to ensure it uses the same one
            # For example, by setting `ocr_engine` parameter in PPStructure if available or letting it pick up defaults.
            # For now, let's let PPStructure manage its OCR component based on lang='en' if it initializes one.
            # The primary OCR is done by self.ocr_engine.
            # If PPStructure's internal OCR needs specific models, it usually defaults well with lang='en'.
        )
        print("PPStructure engine initialized for layout analysis.")


    def _preprocess_image(self, image_bytes):
        try:
            image = Image.open(image_bytes).convert("RGB")
            return np.array(image)
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None

    def _extract_text_from_ocr_results(self, result_data):
        if not result_data or not result_data[0]: # Check if result_data[0] is None or empty
            return ""
        
        all_text = []
        # result_data is typically [[line1_info, line2_info, ...]]
        # line_info is [bbox, (text, confidence)]
        for line_info in result_data[0]: # Iterate through the first (and usually only) element
            if line_info and len(line_info) == 2 and isinstance(line_info[1], tuple):
                text, confidence = line_info[1]
                all_text.append(text)
        return "\n".join(all_text) # Your challenge asks for a single string transcript. "\n" might be okay or " "
                                     # The challenge output is a list of strings, one per image.
                                     # This helper should probably return the single transcript for *one* image.

    def _extract_text_from_layout_results(self, layout_regions, img_for_ocr):
        full_text_parts = []
        if not layout_regions:
            return ""
        
        # Sort regions by top-to-bottom, then left-to-right
        layout_regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

        for i, region in enumerate(layout_regions):
            # We are interested in 'text' type regions primarily for transcription
            if region.get('type') != 'text': # Example: filter for text, table, title. Adjust as needed.
                                             # Or, if your challenge is just "transcribe everything", don't filter by type.
                # For now, let's assume we try to OCR all region types returned by layout
                # and concatenate them.
                pass


            x1, y1, x2, y2 = map(int, region['bbox'])
            # Ensure valid crop dimensions
            if x1 >= x2 or y1 >= y2:
                print(f"Skipping invalid crop box for region {i}: {[x1, y1, x2, y2]}")
                continue
            
            cropped_img_pil = Image.fromarray(img_for_ocr).crop((x1, y1, x2, y2))
            
            # Ensure cropped image is not empty
            if cropped_img_pil.width == 0 or cropped_img_pil.height == 0:
                print(f"Skipping empty cropped image for region {i}")
                continue
                
            cropped_img_np = np.array(cropped_img_pil)

            # Additional check for very small or degenerate crops
            if cropped_img_np.size == 0 or cropped_img_np.shape[0] < 5 or cropped_img_np.shape[1] < 5: # Adjusted threshold
                print(f"Skipping too small or degenerate crop for region {i}")
                continue

            # Use the main OCR engine for consistency
            region_ocr_result = self.ocr_engine.ocr(cropped_img_np, cls=True) 
            
            if region_ocr_result and region_ocr_result[0] is not None:
                region_text_parts = []
                for line_data in region_ocr_result[0]: # line_data is [bbox, (text, confidence)]
                    if line_data and len(line_data) == 2 and isinstance(line_data[1], tuple):
                         text, _ = line_data[1]
                         region_text_parts.append(text)
                if region_text_parts:
                    full_text_parts.append(" ".join(region_text_parts)) # Join words in a region with spaces
        return "\n".join(full_text_parts) # Join text from different regions with newlines

    # Method renamed from perform_ocr to ocr
    def ocr(self, image_bytes, use_layout_analysis=False):
        img_np = self._preprocess_image(image_bytes)
        if img_np is None:
            # Return structure expected by the challenge: {"predictions": ["error message"]}
            # However, ocr_server.py handles raising HTTPException, so this can be simpler.
            return {"error": "Failed to preprocess image"}


        # The challenge expects a single string transcript for each image.
        extracted_text = ""

        if use_layout_analysis:
            print("Performing OCR with Layout Analysis...")
            # Ensure img_np is contiguous for PPStructure if it's sensitive to that
            img_for_structure = np.ascontiguousarray(img_np)
            layout_result_list = self.structure_engine(img_for_structure.copy(), return_ocr_result_in_bbox=False) 
            
            # layout_result_list is a list of dictionaries, each representing a detected region
            if layout_result_list:
                # Filter regions if necessary, e.g., only 'text', 'title', 'table'
                # For now, assume all relevant regions are processed by _extract_text_from_layout_results
                # The example filter: text_regions = [res for res in layout_result_list if res.get('type') not in ['figure', 'table_caption', 'reference']]
                # For the challenge, you probably want to extract text from most regions.
                # Let's pass all regions and let _extract_text_from_layout_results handle them.
                extracted_text = self._extract_text_from_layout_results(layout_result_list, img_np)
                # The challenge output doesn't include layout_boxes, just the transcript.
                # return {"text": extracted_text, "layout_boxes": layout_result_list} 
                return {"text": extracted_text}
            else:
                print("Layout analysis did not return any regions. Falling back to direct OCR.")
                ocr_result = self.ocr_engine.ocr(img_np, cls=True)
                extracted_text = self._extract_text_from_ocr_results(ocr_result)
                return {"text": extracted_text}
        else:
            print("Performing direct OCR without Layout Analysis...")
            ocr_result = self.ocr_engine.ocr(img_np, cls=True)
            extracted_text = self._extract_text_from_ocr_results(ocr_result)
            return {"text": extracted_text}
