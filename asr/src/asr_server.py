"""Runs the ASR server with integrated text autocompletion."""

import base64
from fastapi import FastAPI, Request, HTTPException
import logging # Added for better logging

# Import both classes from your asr_manager.py file
from asr_manager import ASRManager, TextAutoCompleter

# --- Global Instantiation of Models ---
# These are created once when the Uvicorn server starts.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Initializing ASRManager...")
# ASRManager will use MODEL_DIR env var by default as set in Dockerfile
asr_manager_instance = ASRManager()
logging.info("ASRManager initialized.")

logging.info("Initializing TextAutoCompleter...")
# You can change "gpt2" to another compatible model if desired
text_completer_instance = TextAutoCompleter(model_name="gpt2")
logging.info("TextAutoCompleter initialized.")
# --- End Global Instantiation ---

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Models pre-loaded.")
    # Optional: Add a "warm-up" call here if you notice the very first request is slow,
    # especially if torch.compile is used in ASRManager.
    # This is a very basic example; create proper silent audio bytes if you do this.
    try:
        logging.info("Warming up ASR model (optional)...")
        # Create a tiny piece of silent audio for warmup if needed
        # import soundfile as sf
        # import numpy as np
        # import io
        # silent_audio_data = np.zeros(16000 // 10, dtype=np.float32) # 0.1 sec silence
        # byte_io = io.BytesIO()
        # sf.write(byte_io, silent_audio_data, 16000, format='WAV', subtype='PCM_16') # PCM_16 is common
        # warmup_bytes = byte_io.getvalue()
        # asr_manager_instance.asr(warmup_bytes, max_new_asr_tokens=2) # Use the global instance
        
        # logging.info("Warming up TextAutoCompleter (optional)...")
        # text_completer_instance.autocomplete("Hello", max_length=5) # Use the global instance
        logging.info("Model warmup (if enabled) complete.")
    except Exception as e:
        logging.error(f"Error during optional model warmup: {e}")


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files, with optional text autocompletion.

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

    predictions = []
    for i, instance in enumerate(inputs_json["instances"]):
        if "b64" not in instance:
            logging.warning(f"Instance {i} missing 'b64' field. Skipping.")
            predictions.append("Error: Missing 'b64' field in instance.") # Placeholder for skipped
            continue

        try:
            # Reads the base-64 encoded audio and decodes it into bytes.
            audio_bytes = base64.b64decode(instance["b64"])
            if not audio_bytes:
                logging.warning(f"Instance {i} provided empty audio after base64 decode. Skipping.")
                predictions.append("Error: Empty audio data after decoding.")
                continue

            # --- Step 1: Perform partial ASR ---
            # Adjust max_new_asr_tokens as needed for your desired balance.
            # Lower values are faster for ASR but give less context to the completer.
            max_asr_tokens = 15 # Example value, tune this
            partial_transcription = asr_manager_instance.asr(audio_bytes, max_new_asr_tokens=max_asr_tokens)
            logging.info(f"Instance {i}: Partial ASR (first {max_asr_tokens} tokens): '{partial_transcription}'")

            if partial_transcription.startswith("Error:"):
                logging.warning(f"Instance {i}: ASR error: {partial_transcription}")
                predictions.append(partial_transcription) # Propagate ASR error
                continue

            # --- Step 2: Autocomplete the partial transcription ---
            final_transcription = partial_transcription # Default to partial if no completion needed or error

            # Only attempt completion if ASR produced some non-error output.
            # You might add more sophisticated logic here (e.g., based on length).
            if partial_transcription:
                # Adjust max_length for the autocompleter. This is the *total* desired length.
                completion_max_length = len(partial_transcription.split()) + 20 # Example: allow 20 more words
                if completion_max_length < 10: completion_max_length = 10 # ensure some minimum length
                if completion_max_length > 100: completion_max_length = 100 # and some maximum

                logging.info(f"Instance {i}: Attempting to autocomplete with max_length={completion_max_length}")
                final_transcription = text_completer_instance.autocomplete(
                    partial_transcription,
                    max_length=completion_max_length
                )
                logging.info(f"Instance {i}: Final transcription after completion: '{final_transcription}'")
            else:
                logging.info(f"Instance {i}: Skipping autocompletion as partial ASR was empty.")

            predictions.append(final_transcription)

        except base64.binascii.Error as b64_err:
            logging.error(f"Instance {i}: Base64 decoding error: {b64_err}")
            predictions.append("Error: Invalid base64 audio data.")
        except Exception as e:
            logging.error(f"Instance {i}: Unexpected error during processing: {e}", exc_info=True)
            predictions.append(f"Error: Internal server error during processing instance {i}.")

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    # You could add more sophisticated health checks here,
    # e.g., try a very quick dummy inference on models if they have a quick health check method.
    return {"message": "health ok"}

# To run this locally (outside Docker, assuming models and asr_manager.py are in correct relative paths):
# Make sure MODEL_DIR is set in your environment, or ASRManager is modified to find model locally without it.
# uvicorn src.asr_server:app --reload --port 5001 --host 0.0.0.0
# (Adjust path `src.asr_server` if your file structure is different)