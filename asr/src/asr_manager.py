"""Manages the ASR model and includes Text Autocompletion"""

import io
import os
import torch
import librosa
import soundfile as sf
import numpy as np
import logging

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BitsAndBytesConfig,
    GPT2LMHeadModel,    # Added for TextAutoCompleter
    GPT2Tokenizer      # Added for TextAutoCompleter
)
# Note: hf_logging was used in the standalone TextAutoCompleter, 
# you might want to manage its verbosity here too if needed.
# from transformers import logging as hf_logging 
# hf_logging.set_verbosity_error() 


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR_ASR_MANAGER = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# ASRManager Class Definition (as previously modified)
# -----------------------------------------------------------------------------
class ASRManager:
    def __init__(self, model_path_override: str = None):
        # ... (all your ASRManager __init__ code remains here)
        # Make sure `default_model_name_from_training` is correctly set for your ASR model
        default_model_name_from_training = "asr_finetuned_model_small_aug" # Or your actual ASR model name
        # ... (rest of __init__)
        logging.info(f"ASRManager: Initialization complete. Model is on device and in eval mode.")


    def asr(self, audio_bytes: bytes, max_new_asr_tokens: int = None) -> str:
        # ... (all your ASRManager asr method code remains here)
        # ...
        logging.debug("Generating transcription...")
        try:
            self.model.eval()
            with torch.no_grad():
                generation_kwargs = {
                    "num_beams": 1,
                    "do_sample": False
                }
                if max_new_asr_tokens is not None and isinstance(max_new_asr_tokens, int) and max_new_asr_tokens > 0:
                    generation_kwargs["max_new_tokens"] = max_new_asr_tokens
                    logging.info(f"ASR will be limited to {max_new_asr_tokens} new tokens.")
                
                predicted_ids = self.model.generate(self.processor(audio_bytes, sampling_rate=16000, return_tensors="pt").input_features.to(self.device), **generation_kwargs) # Simplified for brevity
            # ... (rest of asr method)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logging.info(f"Partial/Full Transcription: '{transcription}'")
        except Exception as e:
            logging.error(f"Error during model generation: {e}", exc_info=True)
            return "Error: Could not generate transcription."
        return transcription.strip()

# -----------------------------------------------------------------------------
# TextAutoCompleter Class Definition (integrated into this file)
# -----------------------------------------------------------------------------
class TextAutoCompleter:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initializes the TextAutoCompleter.
        Args:
            model_name (str): Name of the pre-trained GPT-2 model to use.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"TextAutoCompleter: Using device: {self.device}")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.model.config.eos_token_id
            logging.info(f"TextAutoCompleter: Loaded model {model_name} successfully.")
        except Exception as e:
            logging.error(f"TextAutoCompleter: Error loading model {model_name}: {e}", exc_info=True)
            raise

    def autocomplete(self, partial_text: str, max_length: int = 50, num_beams: int = 3) -> str:
        """
        Auto-completes the given partial text.
        Args:
            partial_text (str): The text to autocomplete.
            max_length (int): The maximum length of the generated sequence (partial_text + completion).
            num_beams (int): Number of beams for beam search.
        Returns:
            str: The auto-completed text.
        """
        if not partial_text:
            return ""
        logging.debug(f"TextAutoCompleter: Received partial text: '{partial_text}'")
        try:
            inputs = self.tokenizer.encode(partial_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            if max_length <= inputs.shape[1]:
                logging.warning("TextAutoCompleter: max_length is too short for completion, returning original text.")
                return partial_text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            completed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"TextAutoCompleter: Completed text: '{completed_text}'")
            return completed_text.strip()
        except Exception as e:
            logging.error(f"TextAutoCompleter: Error during autocomplete: {e}", exc_info=True)
            return partial_text

# -----------------------------------------------------------------------------
# Main block for testing (if you want to keep it in this file)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running ASRManager & TextAutoCompleter standalone test (integrated file)...")
    
    # --- ASRManager Test ---
    # Ensure this model name and path logic matches your setup
    asr_model_name_for_testing = "asr_finetuned_model_small_aug"
    constructed_asr_model_path = os.path.join(
        SCRIPT_DIR_ASR_MANAGER, "..", "..", "models", # Adjust if your 'models' dir is elsewhere
        asr_model_name_for_testing,
        "model_and_processor_files"
    )
    if not os.path.isdir(constructed_asr_model_path):
         # Try alternative path if script is in src/asr/ and models is sibling to asr/
        constructed_asr_model_path = os.path.join(
            SCRIPT_DIR_ASR_MANAGER, "..", "models",
            asr_model_name_for_testing,
            "model_and_processor_files"
        )


    print(f"Attempting to load ASR model for testing from: {constructed_asr_model_path}")

    if not os.path.isdir(constructed_asr_model_path):
        print(f"ERROR: ASR Model directory not found: {constructed_asr_model_path}")
    else:
        try:
            asr_manager_instance = ASRManager(model_path_override=constructed_asr_model_path)
            print("ASRManager initialized successfully for testing.")

            dummy_sr = 16000
            dummy_duration = 5 
            dummy_audio_data = np.random.randn(dummy_sr * dummy_duration).astype(np.float32)
            
            dummy_audio_bytes_io = io.BytesIO()
            sf.write(dummy_audio_bytes_io, dummy_audio_data, dummy_sr, format='WAV', subtype='FLOAT')
            dummy_audio_bytes_content = dummy_audio_bytes_io.getvalue()
            
            print("\nTesting ASR with limited tokens (e.g., 10)...")
            partial_transcription = asr_manager_instance.asr(dummy_audio_bytes_content, max_new_asr_tokens=10)
            print(f"Test ASR Transcription (partial, 10 tokens): '{partial_transcription}'")

            # --- TextAutoCompleter Test (using the partial ASR output) ---
            if partial_transcription and not partial_transcription.startswith("Error:"):
                print("\nInitializing TextAutoCompleter...")
                completer_instance = TextAutoCompleter() # Uses "gpt2" by default
                print("TextAutoCompleter initialized.")
                
                print(f"\nAttempting to autocomplete ASR output: '{partial_transcription}'")
                final_text = completer_instance.autocomplete(partial_transcription, max_length=30)
                print(f"ASR + Autocomplete: '{final_text}'")
            else:
                print("\nSkipping autocompletion due to ASR error or empty output.")

        except Exception as e:
            print(f"An error occurred during standalone testing: {e}")
            import traceback
            traceback.print_exc()