# src/ocr_manager.py
# Uses standard pre-trained en_PP-OCRv4_rec_infer and its own en_dict.txt
# Output structure strictly matches the user's "working version" (left_col\n\nright_col)

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager (Std Rec Model + Std Dict, Strict Output Format)...")
        
        models_base_dir = os.getenv('MODELS_BASE_DIR')
        if not models_base_dir:
            print("CRITICAL ERROR: MODELS_BASE_DIR environment variable not set.")
            raise EnvironmentError("MODELS_BASE_DIR not set, cannot locate models.")

        det_model_dir = os.path.join(models_base_dir, os.getenv('DET_MODEL_SUBDIR', 'det/en/en_PP-OCRv3_det_infer'))
        
        rec_model_env_subdir = os.getenv('REC_MODEL_SUBDIR', 'rec/en/en_PP-OCRv4_rec_infer') 
        rec_model_dir = os.path.join(models_base_dir, rec_model_env_subdir)
        
        cls_model_dir = os.path.join(models_base_dir, os.getenv('CLS_MODEL_SUBDIR', 'cls/en/ch_ppocr_mobile_v2.0_cls_infer'))
        layout_model_dir = os.path.join(models_base_dir, os.getenv('LAYOUT_MODEL_SUBDIR', 'layout/en/picodet_lcnet_x1_0_fgd_layout_infer'))

        standard_dict_path = os.path.join(rec_model_dir, 'en_dict.txt')

        print(f"Constructed Detection model path: {det_model_dir}")
        print(f"Constructed Recognition model path: {rec_model_dir} (Standard Pre-trained)")
        print(f"Constructed Classification model path: {cls_model_dir}")
        print(f"Constructed Layout model path: {layout_model_dir}")
        print(f"Using Standard Character Dictionary path: {standard_dict_path}")

        key_file = 'inference.pdmodel'
        for model_name, model_path in [
            ("Detection", det_model_dir),
            ("Recognition", rec_model_dir),
            ("Classification", cls_model_dir),
            ("Layout", layout_model_dir)
        ]:
            if not os.path.exists(os.path.join(model_path, key_file)):
                 print(f"WARNING: {model_name} model file '{key_file}' not found at {model_path}")
            else:
                 print(f"SUCCESS: Found {model_name} model file at {model_path}")

        if not os.path.exists(standard_dict_path):
            print(f"CRITICAL WARNING: Standard dictionary NOT FOUND at {standard_dict_path}")
        else:
            print(f"SUCCESS: Found standard dictionary at {standard_dict_path}")

        self.ocr_engine = PaddleOCR(
            use_angle_cls=False, # Matching user's base version
            lang='en',
            layout=True,         # Retained from user's base version
            show_log=True, 
            use_gpu=True,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            rec_char_dict_path=standard_dict_path, # Standard dictionary
            cls_model_dir=cls_model_dir,
            layout_model_dir=layout_model_dir
        )
        print("PaddleOCR engine (Std Rec Model + Std Dict) initialized successfully.")

    def _separate_columns(self, all_text_lines, image_width_px):
        if not all_text_lines:
            return "", ""
        
        if image_width_px <= 0:
            # print("Warning: image_width_px is not positive in _separate_columns. Returning empty columns.")
            return "", ""

        column_divider_x = image_width_px / 2
        left_column_items, right_column_items = [], []
        for line_info in all_text_lines:
            # Robustness checks for line_info structure
            if not (isinstance(line_info, list) and len(line_info) == 2):
                continue 
            points, text_tuple = line_info
            # CORRECTED LINE 81:
            if not (isinstance(points, list) and len(points) > 0 and isinstance(points[0], list) and len(points[0]) == 2):
                continue
            if not (isinstance(text_tuple, tuple) and len(text_tuple) == 2):
                continue
            
            text = text_tuple[0]
            
            try:
                line_center_x = sum(p[0] for p in points) / len(points)
                y_coordinate = points[0][1] 
            except (TypeError, IndexError, ZeroDivisionError):
                continue # Skip line if points are malformed

            if line_center_x < column_divider_x:
                left_column_items.append((text, y_coordinate))
            else:
                right_column_items.append((text, y_coordinate))
        
        left_column_items.sort(key=lambda item: item[1])
        right_column_items.sort(key=lambda item: item[1])
        
        return "\n".join([i[0] for i in left_column_items]), "\n".join([i[0] for i in right_column_items])

    def ocr(self, image_bytes: bytes) -> str:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None: 
                print("OCRManager Error: Could not decode image from bytes.")
                return "Error: Could not decode image."
            
            image_height, image_width, _ = img_np.shape
            if image_height <= 0 or image_width <= 0:
                print("OCRManager Error: Decoded image has invalid dimensions.")
                return "Error: Decoded image has invalid dimensions."

            ocr_result_batch = self.ocr_engine.ocr(img_np, cls=False) # cls=False from user's base
            
            if not ocr_result_batch or not ocr_result_batch[0]: 
                print("OCRManager: OCR returned no results or empty result for the first image.")
                return "" # Return empty string, similar to what "{}\n\n{}".format("","") would be
            
            raw_ocr_data = ocr_result_batch[0] 
            
            processed_lines_for_separation = []
            # If layout=True gives structured blocks (list of dicts), flatten them.
            # Otherwise, assume raw_ocr_data is already a list of lines.
            if raw_ocr_data and isinstance(raw_ocr_data, list) and \
               len(raw_ocr_data) > 0 and isinstance(raw_ocr_data[0], dict) and 'res' in raw_ocr_data[0]:
                # print("OCRManager: Detected structured layout output, flattening for _separate_columns.")
                for block in raw_ocr_data:
                    if block.get('res') and isinstance(block['res'], list):
                        processed_lines_for_separation.extend(block['res'])
            elif raw_ocr_data and isinstance(raw_ocr_data, list): # If it's already a flat list of lines
                processed_lines_for_separation = raw_ocr_data
            # If raw_ocr_data is not in a recognized format, processed_lines_for_separation remains empty.

            left_column_text, right_column_text = self._separate_columns(processed_lines_for_separation, image_width)
            
            # Strict adherence to the "working version's" output format:
            full_text = f"{left_column_text}\n\n{right_column_text}"
            
            return full_text.strip() # .strip() will remove leading/trailing newlines if both columns are empty.
                                     # If one column has text and other is empty, it preserves the \n\n.

        except Exception as e:
            print(f"Error in OCRManager.ocr: {e}")
            import traceback 
            traceback.print_exc()
            return f"Error during OCR processing: {str(e)}"
