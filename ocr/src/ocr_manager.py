# ocr_manager.py
import os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, PPStructure
import yaml # PyYAML

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager...")

        # --- Model Path Configuration from Environment Variables ---
        self.models_base_dir = os.getenv('MODELS_BASE_DIR', '/opt/paddleocr_models')
        
        # Detection Model
        self.det_model_dir = os.path.join(self.models_base_dir, os.getenv('DET_MODEL_SUBDIR', 'det/en/en_PP-OCRv3_det_infer'))
        
        # Recognition Model (Points to your fine-tuned model's directory)
        self.rec_model_dir = os.path.join(self.models_base_dir, os.getenv('REC_MODEL_SUBDIR', 'rec/en/en_PP-OCRv4_rec_infer')) 
        
        # Classification Model
        self.cls_model_dir = os.path.join(self.models_base_dir, os.getenv('CLS_MODEL_SUBDIR', 'cls/en/ch_ppocr_mobile_v2.0_cls_infer'))
        
        # Layout Model
        self.layout_model_dir = os.path.join(self.models_base_dir, os.getenv('LAYOUT_MODEL_SUBDIR', 'layout/en/picodet_lcnet_x1_0_fgd_layout_infer'))

        self.rec_char_dict_path = os.path.join(self.models_base_dir, 'dicts/custom_char_dict.txt')

        print(f"  Models Base Dir: {self.models_base_dir}")
        print(f"  Det Model Dir: {self.det_model_dir}")
        print(f"  Rec Model Dir: {self.rec_model_dir}")
        print(f"  Cls Model Dir: {self.cls_model_dir}")
        print(f"  Layout Model Dir: {self.layout_model_dir}")
        print(f"  Rec Char Dict Path: {self.rec_char_dict_path}")

        if not os.path.exists(self.det_model_dir):
            print(f"WARNING: Detection model directory not found: {self.det_model_dir}")
        if not os.path.exists(self.rec_model_dir):
            print(f"WARNING: Recognition model directory not found: {self.rec_model_dir}. Make sure your fine-tuned model is copied here.")
        if not os.path.exists(self.rec_char_dict_path):
            print(f"WARNING: Character dictionary not found: {self.rec_char_dict_path}")

        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='en', 
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            rec_char_dict_path=self.rec_char_dict_path,
            cls_model_dir=self.cls_model_dir,
            use_gpu=True,
            show_log=True
        )
        print("PaddleOCR engine initialized.")

        self.structure_engine = PPStructure(
            layout_model_dir=self.layout_model_dir,
            show_log=True,
            use_gpu=True,
            layout_dict_path=None
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
        if not result_data:
            return ""
        
        all_text = []
        for line_idx, line_data in enumerate(result_data):
            if line_data: 
                for box_info in line_data:
                    if isinstance(box_info, list) and len(box_info) == 2 and isinstance(box_info[1], tuple):
                        text, confidence = box_info[1]
                        all_text.append(text)
                    elif isinstance(box_info, tuple) and len(box_info) == 2:
                         text, confidence = box_info
                         all_text.append(text)
        return "\n".join(all_text)

    def _extract_text_from_layout_results(self, layout_regions, img_for_ocr):
        full_text_parts = []
        if not layout_regions:
            return ""
        layout_regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

        for i, region in enumerate(layout_regions):
            x1, y1, x2, y2 = map(int, region['bbox'])
            cropped_img_pil = Image.fromarray(img_for_ocr).crop((x1, y1, x2, y2))
            cropped_img_np = np.array(cropped_img_pil)

            if cropped_img_np.size == 0 or cropped_img_np.shape[0] < 10 or cropped_img_np.shape[1] < 10:
                continue

            region_ocr_result = self.ocr_engine.ocr(cropped_img_np, cls=True) 
            
            if region_ocr_result:
                region_text_parts = []
                for line_data in region_ocr_result:
                    if line_data:
                        for box_info in line_data: 
                             text, _ = box_info[1]
                             region_text_parts.append(text)
                if region_text_parts:
                    full_text_parts.append(" ".join(region_text_parts))
        return "\n".join(full_text_parts)

    # Method renamed from perform_ocr to ocr
    def ocr(self, image_bytes, use_layout_analysis=False):
        img_np = self._preprocess_image(image_bytes)
        if img_np is None:
            return {"error": "Failed to preprocess image"}

        if use_layout_analysis:
            print("Performing OCR with Layout Analysis...")
            layout_result = self.structure_engine(img_np.copy(), return_ocr_result_in_bbox=False) 
            if layout_result:
                text_regions = [res for res in layout_result if res.get('type') not in ['figure', 'table_caption', 'reference']]
                extracted_text = self._extract_text_from_layout_results(text_regions, img_np)
                return {"text": extracted_text, "layout_boxes": layout_result} 
            else:
                print("Layout analysis did not return any regions. Falling back to direct OCR.")
                ocr_result = self.ocr_engine.ocr(img_np, cls=True)
                extracted_text = self._extract_text_from_ocr_results(ocr_result)
                return {"text": extracted_text, "layout_boxes": []}
        else:
            print("Performing direct OCR without Layout Analysis...")
            ocr_result = self.ocr_engine.ocr(img_np, cls=True)
            extracted_text = self._extract_text_from_ocr_results(ocr_result)
            return {"text": extracted_text}
