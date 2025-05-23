import os
from paddleocr import PaddleOCR
#from paddleocr.utils.visualizer import draw_ocr
import numpy as np
from PIL import Image

# ——— Paths ———
LABELS_CSV_PATH   = os.path.expanduser("~/til-25-data-chefs/ocr/src/labels.csv")
IMAGE_BASE_DIR    = os.path.expanduser("~/advanced/ocr")
OUTPUT_RESULTS_CSV = os.path.expanduser("~/til-25-data-chefs/ocr/paddleocr_results.csv")

# ——— Initialize PaddleOCR ———
# use angle classification, use GPU if available
ocr = PaddleOCR(use_angle_cls=True, lang="en")  
result = ocr.ocr("path/to/image.jpg", cls=True)

def ocr_image(image_path):
    """Run PaddleOCR on one image, return joined text."""
    result = ocr.ocr(image_path, cls=True)
    # result is list of [ [bbox, (text, confidence)], ... ] per line
    lines = [line[1][0] for line in result[0]]
    return " ".join(lines)

def main():
    # 1. Load your CSV
    df = pd.read_csv(LABELS_CSV_PATH)
    assert "image_path" in df.columns, "CSV needs an 'image_path' column."

    results = []
    for idx, row in df.iterrows():
        img_fname = row["image_path"]
        img_path = os.path.join(IMAGE_BASE_DIR, img_fname)

        if not os.path.exists(img_path):
            print(f"⚠️ Missing: {img_path}")
            recognized = ""
        else:
            recognized = ocr_image(img_path)

        print(f"{img_fname}  →  {recognized}")
        results.append({
            "image_path": img_fname,
            "ground_truth": row.get("text", ""),
            "paddleocr_text": recognized
        })

    # 2. Save results for later analysis
    pd.DataFrame(results).to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"Results written to {OUTPUT_RESULTS_CSV}")

if __name__ == "__main__":
    main()
