import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
import jiwer
from typing import Any
import requests
from dotenv import load_dotenv
import itertools
from tqdm import tqdm


load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4
# Define how many samples you want to test with
NUM_SAMPLES_TO_TEST = 10 # <--- ADD THIS LINE


cer_transforms = jiwer.Compose([
    jiwer.SubstituteRegexes({"-": ""}),
    jiwer.RemoveWhiteSpace(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / instance["document"], "rb") as file:
            document_bytes = file.read()
        yield {
            **instance,
            "b64": base64.b64encode(document_bytes).decode("ascii"),
        }


def score_ocr(preds: Sequence[str], ground_truth: Sequence[str]) -> float:
    return 1 - jiwer.cer(
        ground_truth,
        preds,
        truth_transform=cer_transforms,
        hypothesis_transform=cer_transforms,
    )


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/ocr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "ocr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    # --- MODIFICATION START ---
    # Slice the instances list to use only a subset for testing
    if NUM_SAMPLES_TO_TEST > 0 and NUM_SAMPLES_TO_TEST < len(instances):
        print(f"Using a subset of {NUM_SAMPLES_TO_TEST} samples for testing.")
        instances_to_process = instances[:NUM_SAMPLES_TO_TEST]
    else:
        print(f"Using all {len(instances)} samples.")
        instances_to_process = instances
    # --- MODIFICATION END ---

    # Use instances_to_process for the generator
    batch_generator = itertools.batched(sample_generator(instances_to_process, data_dir), n=BATCH_SIZE)

    results = []
    # Adjust tqdm total based on the subset
    for batch in tqdm(batch_generator, total=math.ceil(len(instances_to_process) / BATCH_SIZE)):
        response = requests.post("http://localhost:5003/ocr", data=json.dumps({
            "instances": batch, # This 'batch' will now be from the subset
        }))
        response_data = response.json()
        if "predictions" in response_data:
            results.extend(response_data["predictions"])
        else:
            print(f"Warning: 'predictions' key not found in response for a batch. Response: {response_data}")
            # Add empty strings or handle as appropriate if predictions are missing
            results.extend(["ERROR_NO_PREDICTION"] * len(batch))


    results_path = results_dir / "ocr_results_sample.json" # Consider a different name for sample results
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)
    
    ground_truths = []
    # Use instances_to_process to get corresponding ground truths
    for instance in instances_to_process:
        with open(data_dir / instance["contents"], "r") as file:
            document_contents = file.read().strip()
        ground_truths.append(document_contents)

    # --- MODIFICATION FOR CHECKING OUTPUT ---
    print("\n--- Sample Predictions vs Ground Truths ---")
    for i in range(min(len(results), len(ground_truths), NUM_SAMPLES_TO_TEST)): # Display up to NUM_SAMPLES_TO_TEST
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {ground_truths[i][:200]}{'...' if len(ground_truths[i]) > 200 else ''}") # Print first 200 chars
        print(f"  Prediction  : {results[i][:200]}{'...' if len(results[i]) > 200 else ''}")   # Print first 200 chars
    print("-----------------------------------------\n")
    # --- END MODIFICATION FOR CHECKING OUTPUT ---

    if not ground_truths:
        print("No ground truths to score against.")
    elif len(results) != len(ground_truths):
        print(f"Warning: Mismatch in number of results ({len(results)}) and ground truths ({len(ground_truths)}). CER might be inaccurate.")
        # Optionally, you might want to skip scoring or handle this.
        # For now, let's try to score with what we have, jiwer might handle length mismatches.
        # Or, ensure lengths match before scoring:
        min_len = min(len(results), len(ground_truths))
        score = score_ocr(results[:min_len], ground_truths[:min_len])
        print(f"1 - CER (calculated on {min_len} pairs due to length mismatch): {score}")
    else:
        score = score_ocr(results, ground_truths)
        print("1 - CER:", score)


if __name__ == "__main__":
    main()