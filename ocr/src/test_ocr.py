# test_ocr.py
import requests
import json
import base64
import time
import os
from jiwer import cer # Character Error Rate

# --- Configuration ---
OCR_SERVER_URL = "http://localhost:5003/ocr" # Your server URL
IMAGE_LIST_FILE_ABSOLUTE_PATH = "/home/jupyter/advanced/ocr/ocr.jsonl" 
NUM_IMAGES_TO_PROCESS = 50 
OUTPUT_RESULTS_FILE_TEMPLATE = "/home/jupyter/data-chefs/ocr_results_first_{num_images}.json"
# USE_LAYOUT_ANALYSIS is not sent to the server as per your ocr_server.py model

# --- Helper Functions ---
def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def load_ground_truth(gt_path):
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found at {gt_path}")
        return None
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading ground truth {gt_path}: {e}")
        return None

# --- Main Script ---
if __name__ == "__main__":
    print(f"Processing the first {NUM_IMAGES_TO_PROCESS} images.")
    
    image_list_file_path = IMAGE_LIST_FILE_ABSOLUTE_PATH
    output_results_file_path = OUTPUT_RESULTS_FILE_TEMPLATE.format(num_images=NUM_IMAGES_TO_PROCESS)

    print(f"Attempting to load image list from: {image_list_file_path}")

    if not os.path.exists(image_list_file_path):
        print(f"Error: Image list file not found at {image_list_file_path}")
        exit()

    all_ground_truths = {}
    instances_to_send = [] # List to hold instances for the batched request
    
    base_dir_for_jsonl_paths = os.path.dirname(image_list_file_path)

    with open(image_list_file_path, "r") as f:
        for i, line in enumerate(f):
            if len(instances_to_send) >= NUM_IMAGES_TO_PROCESS: # Limit based on successful preparations
                break
            try:
                record = json.loads(line)
                image_key_original = record.get("key", str(i)) # Keep original key for mapping later
                
                image_filename_from_jsonl = record["document"]
                gt_filename_from_jsonl = record["contents"]

                image_path_absolute = os.path.normpath(os.path.join(base_dir_for_jsonl_paths, image_filename_from_jsonl))
                gt_path_absolute = os.path.normpath(os.path.join(base_dir_for_jsonl_paths, gt_filename_from_jsonl))
                
                image_base64 = encode_image_to_base64(image_path_absolute)
                ground_truth = load_ground_truth(gt_path_absolute)

                if image_base64 and ground_truth is not None:
                    # Prepare instance for server (using "key" and "b64")
                    instances_to_send.append({
                        "key": image_key_original, # Server expects "key"
                        "b64": image_base64    # Server expects "b64"
                    })
                    all_ground_truths[str(image_key_original)] = ground_truth # Store GT by original key
                else:
                    print(f"Skipping image key {image_key_original} due to missing image or ground truth.")

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Skipping line due to missing key {e} in record: {record if 'record' in locals() else 'Error before record assignment'}")

    all_predictions = {} # To store predictions mapped by original key

    if instances_to_send:
        print(f"\nSending {len(instances_to_send)} instance(s) in a single batch to the OCR server...")
        batched_payload = {"instances": instances_to_send}
        
        try:
            response = requests.post(OCR_SERVER_URL, json=batched_payload, timeout=300) # Increased timeout for batch
            response.raise_for_status()
            server_response_data = response.json()
            
            if "predictions" in server_response_data and isinstance(server_response_data["predictions"], list):
                server_predictions_list = server_response_data["predictions"]
                if len(server_predictions_list) == len(instances_to_send):
                    for i, instance_sent in enumerate(instances_to_send):
                        original_key = str(instance_sent["key"]) # Key used to send
                        all_predictions[original_key] = server_predictions_list[i]
                    print("Successfully received and mapped batched predictions.")
                else:
                    print(f"Error: Mismatch in number of predictions received ({len(server_predictions_list)}) and instances sent ({len(instances_to_send)}).")
                    # Fill with error for unmapped predictions
                    for i, instance_sent in enumerate(instances_to_send):
                        original_key = str(instance_sent["key"])
                        all_predictions[original_key] = "Error: Prediction count mismatch"
            else:
                print(f"Error: 'predictions' key not found or not a list in server response: {server_response_data}")
                for i, instance_sent in enumerate(instances_to_send):
                    all_predictions[str(instance_sent["key"])] = "Error: Invalid server response structure"

        except requests.exceptions.RequestException as e:
            print(f"Error sending batched request: {e}")
            for i, instance_sent in enumerate(instances_to_send): # Mark all as failed
                all_predictions[str(instance_sent["key"])] = f"Error: Request failed {e}"
        except json.JSONDecodeError:
            print(f"Error decoding JSON response from server. Response text: {response.text}")
            for i, instance_sent in enumerate(instances_to_send):
                all_predictions[str(instance_sent["key"])] = "Error: Non-JSON server response"
    else:
        print("No instances prepared to send.")


    print(f"\nSaving test results for {len(all_predictions)} processed instances to {output_results_file_path}")
    results_to_save = {
        "predictions": all_predictions, # Predictions mapped by original key
        "ground_truths": all_ground_truths
    }
    try:
        os.makedirs(os.path.dirname(output_results_file_path), exist_ok=True) 
        with open(output_results_file_path, "w") as outfile:
            json.dump(results_to_save, outfile, indent=2)
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    print(f"\n--- Scoring for {len(all_predictions)} processed instances ---")
    print("Predictions mapped from server:")
    valid_keys_for_scoring = [key for key in all_ground_truths if key in all_predictions]

    for key in sorted(valid_keys_for_scoring):
        pred_text = all_predictions.get(key, "PREDICTION_MISSING") # Use .get for safety
        truncated_pred = (pred_text[:70] + '...') if len(pred_text) > 70 else pred_text
        print(f"  Instance Key {key}: '{truncated_pred}' (Length: {len(pred_text)})")

    print("\nGround Truths loaded (for scored instances):")
    for key in sorted(valid_keys_for_scoring):
        gt_text = all_ground_truths.get(key, "GT_MISSING")
        truncated_gt = (gt_text[:70] + '...') if len(gt_text) > 70 else gt_text
        print(f"  Instance Key {key}: '{truncated_gt}' (Length: {len(gt_text)})")
    
    gt_list_for_cer = []
    pred_list_for_cer = []
    
    for key in sorted(valid_keys_for_scoring):
        # Ensure both exist before adding to CER lists
        if key in all_ground_truths and key in all_predictions:
            gt_list_for_cer.append(all_ground_truths[key])
            pred_list_for_cer.append(all_predictions[key]) 

    if not gt_list_for_cer:
        print("\nNo valid ground truth/prediction pairs available for CER calculation.")
    else:
        try:
            error = cer(gt_list_for_cer, pred_list_for_cer)
            accuracy = 1 - error
            print(f"\n1 - CER for processed images: {accuracy}")
        except Exception as e:
            print(f"\nError calculating CER: {e}")