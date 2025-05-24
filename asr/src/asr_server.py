"""Runs the ASR server, using batch processing."""

import base64
from fastapi import FastAPI, Request, HTTPException
import logging
from typing import List # For type hinting

# Assuming asr_manager is in a 'src' subdirectory relative to where this server script might be run from,
# or Python's path is set up to find 'src.asr_manager'.
# If asr_manager.py is in the same directory, it would be: from asr_manager import ASRManager
from asr_manager import ASRManager # Keep this if asr_manager.py is in src/

# Configure logging early and set to DEBUG to see all messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

app = FastAPI()

logging.info("Initializing ASRManager in server module...")
manager = ASRManager() # Instantiated once
logging.info("ASRManager initialized in server module.")


@app.on_event("startup")
async def startup_event():
    logging.info("Application startup event: ASRManager should be pre-loaded.")
    # Optional: Warmup call with a dummy batch (can be useful)
    # try:
    #     logging.info("Warming up ASR model with a dummy batch...")
    #     import numpy as np; import soundfile as sf; import io
    #     dummy_audio = np.zeros(16000 // 2, dtype=np.float32) # 0.5s of silence
    #     byte_io = io.BytesIO(); sf.write(byte_io, dummy_audio, 16000, format='WAV')
    #     # Use a small, valid batch for warmup
    #     manager.asr_batch([byte_io.getvalue(), byte_io.getvalue()])
    #     logging.info("ASR model warmed up.")
    # except Exception as e:
    #     logging.error(f"Error during model warmup: {e}", exc_info=True)


@app.post("/asr")
async def asr(request: Request) -> dict[str, List[str]]:
    """Performs ASR on a batch of audio files.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64. Expected JSON format: {"instances": [{"b64": "base64string"}, ...]}

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
    """
    try:
        inputs_json = await request.json()
    except Exception as e:
        logging.error(f"Invalid JSON received: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

    if "instances" not in inputs_json or not isinstance(inputs_json["instances"], list):
        logging.error("Missing or invalid 'instances' field in JSON.")
        raise HTTPException(status_code=400, detail="JSON must contain a list of 'instances'.")

    audio_bytes_list: List[bytes] = []
    # Keep track of original indices for items that fail b64 or are missing 'b64'
    # This helps construct the final response with errors in the correct places.
    original_indices_for_successful_b64_decode: List[int] = []

    for i, instance in enumerate(inputs_json["instances"]):
        if "b64" not in instance or not isinstance(instance["b64"], str):
            logging.warning(f"Instance {i} missing 'b64' field or not a string. Will be marked as error.")
            continue # This instance won't be processed by ASRManager

        try:
            audio_bytes = base64.b64decode(instance["b64"])
            if not audio_bytes:
                logging.warning(f"Instance {i} provided empty audio after base64 decode. Will be marked as error.")
                continue # This instance won't be processed
            audio_bytes_list.append(audio_bytes)
            original_indices_for_successful_b64_decode.append(i) # Store original index
        except base64.binascii.Error as b64_err:
            logging.warning(f"Instance {i}: Base64 decoding error: {b64_err}. Will be marked as error.")
            # This instance won't be processed
        except Exception as e:
            logging.warning(f"Instance {i}: Unexpected error preparing audio: {e}. Will be marked as error.")
            # This instance won't be processed

    # Initialize final predictions list with error messages for all instances
    predictions = ["Error: Invalid audio data or missing 'b64' field."] * len(inputs_json["instances"])

    if audio_bytes_list: # Only call ASRManager if there's something to process
        logging.debug(f"Sending a batch of {len(audio_bytes_list)} successfully decoded audios to ASRManager.")
        transcriptions_from_manager = manager.asr_batch(audio_bytes_list)

        # Place successful transcriptions back into the correct original positions
        if len(transcriptions_from_manager) == len(original_indices_for_successful_b64_decode):
            for i, original_idx in enumerate(original_indices_for_successful_b64_decode):
                predictions[original_idx] = transcriptions_from_manager[i]
        else:
            logging.error(f"Mismatch between ASR results ({len(transcriptions_from_manager)}) and successfully decoded audios ({len(original_indices_for_successful_b64_decode)}). Filling with errors.")
            # This case indicates an issue, so we ensure all processed items reflect some error from ASR
            for original_idx in original_indices_for_successful_b64_decode:
                predictions[original_idx] = "Error: ASR processing failed or result count mismatch."
    elif len(inputs_json["instances"]) > 0: # No audio was successfully decoded, but instances were provided
        logging.warning("No audio data successfully decoded from any instance.")
        # The predictions list is already filled with errors, so no action needed here.
    else: # No instances provided at all
        logging.info("No instances provided in the request.")
        predictions = []


    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}

# If you run this file directly with uvicorn for testing:
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5001, log_level="debug")