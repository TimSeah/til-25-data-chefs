"""Manages the ASR model"""

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
    BitsAndBytesConfig # Added for quantization
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Determine script's directory for relative path construction ---
SCRIPT_DIR_ASR_MANAGER = os.path.dirname(os.path.abspath(__file__)) # ~/til-25-data-chefs/asr/src

class ASRManager:
    def __init__(self, model_path_override: str = None):
        """
        Initializes the ASRManager, loading the model and processor with optimizations.

        Args:
            model_path_override (str, optional): Explicit path to the model directory.
                                                 If None, uses environment variable or default.
        """
        resolved_model_path = ""
        if model_path_override:
            resolved_model_path = model_path_override
            logging.info(f"Using provided model_path_override: {resolved_model_path}")
        else:
            model_path_env = os.getenv("MODEL_DIR")
            if model_path_env:
                resolved_model_path = model_path_env
                logging.info(f"Using MODEL_DIR from environment variable: {resolved_model_path}")
            else:
                # Fallback to a relative path. This should point to the specific fine-tuned model's
                # "model_and_processor_files" subdirectory.
                default_model_name_from_training = "asr_finetuned_v3_epochs7_cosine_compile" # <<< UPDATE THIS if your training EXPERIMENT_NAME changes
                default_relative_path = os.path.join(
                    SCRIPT_DIR_ASR_MANAGER, "..", "models",
                    default_model_name_from_training,
                    "model_and_processor_files"
                )
                if os.path.isdir(default_relative_path):
                    resolved_model_path = default_relative_path
                    logging.info(f"MODEL_DIR not set, using default relative path: {resolved_model_path}")
                else:
                    alt_default_relative_path = os.path.join(
                        SCRIPT_DIR_ASR_MANAGER, "..", "..", "models",
                        default_model_name_from_training,
                        "model_and_processor_files"
                    )
                    if os.path.isdir(alt_default_relative_path):
                        resolved_model_path = alt_default_relative_path
                        logging.info(f"MODEL_DIR not set, using alternative default relative path: {resolved_model_path}")
                    else:
                        logging.error(f"MODEL_DIR not set, and default paths ('{default_relative_path}', '{alt_default_relative_path}') not found.")
                        raise FileNotFoundError(f"Model directory not found. Set MODEL_DIR or ensure default path is correct.")

        logging.info(f"ASRManager: Final model path to load: {resolved_model_path}")
        if not os.path.isdir(resolved_model_path):
             logging.error(f"ASRManager: Model path is not a valid directory: {resolved_model_path}")
             raise FileNotFoundError(f"Model path is not a valid directory: {resolved_model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"ASRManager: Using device: {self.device}")

        quantization_config_bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logging.info("ASRManager: Configured 4-bit quantization with BitsAndBytes.")

        try:
            self.processor = WhisperProcessor.from_pretrained(resolved_model_path)
            logging.info("ASRManager: WhisperProcessor loaded successfully.")

            logging.info("ASRManager: Attempting to load WhisperForConditionalGeneration model with quantization...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                resolved_model_path,
                quantization_config=quantization_config_bnb,
            )
            logging.info("ASRManager: Model loaded successfully with quantization.")

        except Exception as e:
            logging.error(f"ASRManager: Error loading model with quantization from {resolved_model_path}: {e}. Falling back.", exc_info=True)
            self.processor = WhisperProcessor.from_pretrained(resolved_model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(resolved_model_path)
            logging.warning("ASRManager: Loaded model without quantization due to previous error.")

        if not hasattr(self.model, 'hf_device_map'):
            self.model.to(self.device)
        logging.info(f"ASRManager: Model is on device: {self.model.device if not hasattr(self.model, 'hf_device_map') else 'managed by device_map'}")

        if hasattr(torch, 'compile') and torch.__version__ < "3.0":
            if not hasattr(self.model, 'hf_device_map') or len(self.model.hf_device_map) <= 1:
                logging.info("ASRManager: Attempting to apply torch.compile() to the model...")
                try:
                    # --- MODIFICATION FOR POTENTIAL SPEEDUP ---
                    # Changed mode from "reduce-overhead" to "max-autotune".
                    # "max-autotune" takes longer to compile initially but aims for maximum inference speed.
                    self.model = torch.compile(self.model, mode="max-autotune")
                    # Original: self.model = torch.compile(self.model, mode="reduce-overhead")
                    logging.info("ASRManager: Successfully applied torch.compile() with mode='max-autotune'.")
                except Exception as e:
                    logging.error(f"ASRManager: Failed to apply torch.compile(): {e}", exc_info=True)
            else:
                logging.info("ASRManager: Skipping torch.compile() due to multi-device placement by device_map.")
        else:
            logging.info("ASRManager: torch.compile() not available (requires PyTorch 2.0+ and < 3.0 for this check).")

        if hasattr(self.model.generation_config, 'forced_decoder_ids') and \
           self.model.generation_config.forced_decoder_ids is not None:
            logging.info(f"Found forced_decoder_ids. Clearing them.")
            self.model.generation_config.forced_decoder_ids = None

        tokenizer_lang = getattr(self.processor.tokenizer, 'language', None)
        tokenizer_task = getattr(self.processor.tokenizer, 'task', None)
        self.model.generation_config.language = tokenizer_lang if tokenizer_lang else "english"
        self.model.generation_config.task = tokenizer_task if tokenizer_task else "transcribe"
        logging.info(f"Effective generation_config - Language: {self.model.generation_config.language}, Task: {self.model.generation_config.task}")

        self.model.eval()
        logging.info(f"ASRManager: Initialization complete. Model is on device '{self.model.device if not hasattr(self.model, 'hf_device_map') else 'managed by device_map'}' and in eval mode.")


    def asr(self, audio_bytes: bytes) -> str:
        logging.debug("Received audio bytes for ASR.")
        try:
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio, sr = sf.read(audio_buffer, dtype='float32')
            logging.debug(f"Audio read. Original SR: {sr}, Samples: {len(audio)}, Dims: {audio.ndim}")
        except Exception as e:
            logging.error(f"Error decoding audio bytes: {e}", exc_info=True)
            return "Error: Could not process audio (decode failed)."

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
            logging.debug("Converted stereo to mono.")

        if sr != 16000:
            logging.debug(f"Resampling audio from {sr}Hz to 16000Hz.")
            try:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except Exception as e:
                logging.error(f"Error resampling audio: {e}", exc_info=True)
                return "Error: Could not resample audio."
        
        logging.debug("Processing audio features.")
        try:
            if audio.ndim > 1: audio = audio.flatten()
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            logging.debug("Audio features processed.")
        except Exception as e:
            logging.error(f"Error processing features: {e}", exc_info=True)
            return "Error: Could not process audio features."

        logging.debug("Generating transcription with optimized settings...")
        try:
            self.model.eval() 
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    num_beams=1,
                    do_sample=False
                )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logging.info(f"Transcription: '{transcription}'")
        except Exception as e:
            logging.error(f"Error during model generation: {e}", exc_info=True)
            return "Error: Could not generate transcription."

        return transcription.strip()


if __name__ == '__main__':
    print("Running ASRManager standalone test...")
    
    model_name_for_testing = "asr_finetuned_model_small_aug" # <<< UPDATE THIS
    
    constructed_model_path = os.path.join(
        SCRIPT_DIR_ASR_MANAGER,
        "..",
        "models_OLD",
        model_name_for_testing,
        "model_and_processor_files"
    )
    if not os.path.isdir(constructed_model_path):
        constructed_model_path = os.path.join(
            SCRIPT_DIR_ASR_MANAGER,
            "..",
            "..",
            "models",
            model_name_for_testing,
            "model_and_processor_files"
        )

    print(f"Attempting to load model for testing from: {constructed_model_path}")

    if not os.path.isdir(constructed_model_path):
        print(f"ERROR: Fine-tuned model directory not found at the expected path for testing: {constructed_model_path}")
        print(f"Please ensure '{model_name_for_testing}' exists and contains 'model_and_processor_files'.")
    else:
        try:
            manager = ASRManager(model_path_override=constructed_model_path)
            print("ASRManager initialized successfully for testing.")

            dummy_sr = 16000
            dummy_duration = 1 
            dummy_audio_data = np.zeros(dummy_sr * dummy_duration, dtype=np.float32)
            
            dummy_audio_bytes_io = io.BytesIO()
            sf.write(dummy_audio_bytes_io, dummy_audio_data, dummy_sr, format='WAV', subtype='FLOAT')
            dummy_audio_bytes_content = dummy_audio_bytes_io.getvalue()
            
            print("\nTesting ASR with dummy silent audio...")
            transcription = manager.asr(dummy_audio_bytes_content)
            print(f"Test Transcription (dummy audio): '{transcription}'")
            print("\nIf the transcription for silent audio is empty or has repetitive tokens, it's often normal.")

        except Exception as e:
            print(f"An error occurred during standalone testing: {e}")
            import traceback
            traceback.print_exc()
