"""Manages the ASR model with batch processing capability"""

import io
import os
import torch
import librosa
import soundfile as sf
import numpy as np
import logging
from typing import List

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BitsAndBytesConfig
)

# Ensure root logger is at least INFO for this module if not set by server
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger for this module

SCRIPT_DIR_ASR_MANAGER = os.path.dirname(os.path.abspath(__file__))

class ASRManager:
    def __init__(self, model_path_override: str = None):
        self.quantized_successfully = False
        resolved_model_path = ""
        # Path resolution logic (as before)
        if model_path_override:
            resolved_model_path = model_path_override
            logger.info(f"Using provided model_path_override: {resolved_model_path}")
        else:
            model_path_env = os.getenv("MODEL_DIR")
            if model_path_env:
                resolved_model_path = model_path_env
                logger.info(f"Using MODEL_DIR from environment variable: {resolved_model_path}")
            else:
                default_model_name_from_training = "asr_finetuned_model_small_aug"
                base_model_path_attempt1 = os.path.join(SCRIPT_DIR_ASR_MANAGER, "..", "..", "models", default_model_name_from_training, "model_and_processor_files")
                base_model_path_attempt2 = os.path.join(SCRIPT_DIR_ASR_MANAGER, "..", "models", default_model_name_from_training, "model_and_processor_files")

                if os.path.isdir(base_model_path_attempt1):
                    resolved_model_path = base_model_path_attempt1
                elif os.path.isdir(base_model_path_attempt2):
                    resolved_model_path = base_model_path_attempt2
                else:
                    logger.error(f"MODEL_DIR not set, and default paths ('{base_model_path_attempt1}', '{base_model_path_attempt2}') not found for model '{default_model_name_from_training}'.")
                    raise FileNotFoundError(f"Model directory not found. Set MODEL_DIR or ensure default path is correct.")
                logger.info(f"MODEL_DIR not set, using resolved default path: {resolved_model_path}")

        logger.info(f"ASRManager: Final model path to load: {resolved_model_path}")
        if not os.path.isdir(resolved_model_path):
             logger.error(f"ASRManager: Model path is not a valid directory: {resolved_model_path}")
             raise FileNotFoundError(f"Model path is not a valid directory: {resolved_model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ASRManager: Determined execution device: {self.device}")

        quantization_config_bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("ASRManager: Configured 4-bit quantization with BitsAndBytes.")

        try:
            self.processor = WhisperProcessor.from_pretrained(resolved_model_path)
            logger.info("ASRManager: WhisperProcessor loaded successfully.")
            self.expected_feature_length = self.processor.feature_extractor.n_samples // self.processor.feature_extractor.hop_length
            logger.info(f"ASRManager: Processor's feature_extractor.chunk_length: {getattr(self.processor.feature_extractor, 'chunk_length', 'N/A')}")
            logger.info(f"ASRManager: Processor's feature_extractor.n_samples: {getattr(self.processor.feature_extractor, 'n_samples', 'N/A')} (This is {getattr(self.processor.feature_extractor, 'n_samples', 0)//16000}s)")
            logger.info(f"ASRManager: Calculated expected feature length for model: {self.expected_feature_length}")


            logger.info(f"ASRManager: Attempting to load model with quantization, device_map='auto', torch_dtype=float16, low_cpu_mem_usage=True...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                resolved_model_path,
                quantization_config=quantization_config_bnb,
                device_map="auto",
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True 
            )
            logger.info("ASRManager: Model loaded successfully with quantization.")
            self.quantized_successfully = True

        except Exception as e:
            logger.error(f"ASRManager: Error loading model with quantization: {e}. Falling back.", exc_info=True)
            if not hasattr(self, 'processor') or self.processor is None: 
                 self.processor = WhisperProcessor.from_pretrained(resolved_model_path)
                 self.expected_feature_length = self.processor.feature_extractor.n_samples // self.processor.feature_extractor.hop_length
                 logger.info(f"ASRManager (fallback): Processor's feature_extractor.chunk_length: {getattr(self.processor.feature_extractor, 'chunk_length', 'N/A')}")
                 logger.info(f"ASRManager (fallback): Processor's feature_extractor.n_samples: {getattr(self.processor.feature_extractor, 'n_samples', 'N/A')} (This is {getattr(self.processor.feature_extractor, 'n_samples', 0)//16000}s)")
                 logger.info(f"ASRManager (fallback): Calculated expected feature length for model: {self.expected_feature_length}")


            self.model = WhisperForConditionalGeneration.from_pretrained(resolved_model_path)
            logger.warning("ASRManager: Loaded model without quantization due to previous error.")
            logger.info(f"ASRManager: Moving non-quantized model to device: {self.device}")
            self.model.to(self.device)
            self.quantized_successfully = False

        if self.quantized_successfully:
            logger.info(f"ASRManager: Quantized model device map: {getattr(self.model, 'hf_device_map', 'N/A')}")
            logger.info(f"ASRManager: Quantized model is on device type(s): {set(d.type for d in self.model.hf_device_map.values()) if hasattr(self.model, 'hf_device_map') and isinstance(self.model.hf_device_map, dict) else 'N/A'}")
        else:
            logger.info(f"ASRManager: Non-quantized model is on device: {self.model.device}")

        logger.info("ASRManager: torch.compile() step temporarily skipped for debugging quantization.")
        
        if hasattr(self.model.generation_config, 'forced_decoder_ids') and \
           self.model.generation_config.forced_decoder_ids is not None:
            self.model.generation_config.forced_decoder_ids = None
        
        tokenizer_lang = getattr(self.processor.tokenizer, 'language', None)
        tokenizer_task = getattr(self.processor.tokenizer, 'task', None)
        self.model.generation_config.language = tokenizer_lang if tokenizer_lang else "english"
        self.model.generation_config.task = tokenizer_task if tokenizer_task else "transcribe"

        self.model.eval()
        logger.info("ASRManager: Initialization complete.")

    def _preprocess_audio_bytes(self, audio_bytes: bytes) -> np.ndarray | None:
        try:
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio, sr = sf.read(audio_buffer, dtype='float32')
            if audio.ndim == 2: audio = np.mean(audio, axis=1)
            if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if audio.ndim > 1: audio = audio.flatten()
            # Ensure audio is not excessively short for feature extraction (e.g. min 0.1s)
            # A very short audio might lead to too few frames.
            # Whisper hop length is 160 samples (0.01s), window is 400 samples (0.025s)
            # Need at least enough samples for one frame.
            if audio.size < self.processor.feature_extractor.hop_length * 2: # Heuristic, e.g. min 2 frames
                 logger.warning(f"Preprocessed audio is very short: {audio.size} samples. This might lead to issues.")
            return audio
        except Exception as e:
            logger.error(f"Error decoding/preprocessing audio for an item: {e}", exc_info=False)
            return None

    def asr_batch(self, audio_bytes_list: List[bytes]) -> List[str]:
        if not audio_bytes_list:
            return []

        processed_audios = []
        error_flags = [False] * len(audio_bytes_list) 

        logger.debug(f"Preprocessing batch of {len(audio_bytes_list)} audio items.")
        for i, audio_b in enumerate(audio_bytes_list):
            audio_arr = self._preprocess_audio_bytes(audio_b)
            if audio_arr is not None and audio_arr.size > 0 : # Ensure non-empty after preprocessing
                logger.debug(f"Item {i}: Successfully preprocessed. Raw samples: {audio_arr.size}")
                processed_audios.append(audio_arr)
            else:
                logger.warning(f"Item {i}: Audio could not be processed or resulted in problematic array. Using placeholder.")
                error_flags[i] = True
                placeholder_audio = np.zeros(self.processor.feature_extractor.hop_length * 10, dtype=np.float32) # Placeholder e.g. 0.1s
                processed_audios.append(placeholder_audio)
                logger.debug(f"Item {i}: Placeholder created with {placeholder_audio.size} samples.")
        
        if all(ef for ef in error_flags) and audio_bytes_list:
            logger.warning("All audios in the batch failed preprocessing adequately.")
            return ["Error: Could not process audio (decode/preprocess failed)." for _ in audio_bytes_list]

        logger.debug(f"Calling processor for features with {len(processed_audios)} items.")
        try:
            input_features_batch = self.processor( # Renamed variable for clarity
                audio=processed_audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=self.expected_feature_length, # Use calculated expected length (should be 3000)
                truncation=True,
                return_attention_mask=True
            )
            
            processed_input_features = input_features_batch.input_features 
            attention_mask = input_features_batch.attention_mask

            logger.debug(f"Shape of processed_input_features from processor: {processed_input_features.shape}")
            logger.debug(f"Shape of attention_mask from processor: {attention_mask.shape}")


            if not self.quantized_successfully and self.device.type != 'cpu':
                 processed_input_features = processed_input_features.to(self.device)
                 attention_mask = attention_mask.to(self.device)

            logger.debug("Audio features batch processed and moved to device if necessary.")
        except Exception as e:
            logger.error(f"Error processing features for batch: {e}", exc_info=True)
            final_error_results = []
            for i_original in range(len(audio_bytes_list)):
                if error_flags[i_original]: # If it failed in _preprocess_audio_bytes
                    final_error_results.append("Error: Could not process audio (decode/preprocess failed).")
                else: # If it failed in self.processor or later in this try block
                    final_error_results.append("Error: Could not process audio features for batch.")
            return final_error_results

        logger.debug(f"Generating transcriptions for batch with features of shape: {processed_input_features.shape}")
        try:
            self.model.eval()
            with torch.no_grad():
                predicted_ids_batch = self.model.generate(
                    input_features=processed_input_features, # Corrected 'inputs' to 'input_features'
                    attention_mask=attention_mask,
                    num_beams=1,
                    do_sample=False
                )
            transcriptions_batch = self.processor.batch_decode(predicted_ids_batch, skip_special_tokens=True)
            logger.info(f"Batch transcriptions generated. Count: {len(transcriptions_batch)}")

            final_results = []
            processed_transcription_idx = 0
            for i in range(len(audio_bytes_list)):
                if error_flags[i]: 
                    final_results.append("Error: Could not process audio (decode/preprocess failed).")
                else: 
                    if processed_transcription_idx < len(transcriptions_batch):
                        final_results.append(transcriptions_batch[processed_transcription_idx].strip())
                        processed_transcription_idx += 1
                    else:
                        logger.error(f"Mismatch in transcription results for original index {i}. Expected transcription but not found.")
                        final_results.append("Error: Transcription result missing after processing.")
            return final_results

        except Exception as e:
            logger.error(f"Error during batch model generation: {e}", exc_info=True)
            final_error_results = []
            for i_original in range(len(audio_bytes_list)):
                if error_flags[i_original]:
                    final_error_results.append("Error: Could not process audio (decode/preprocess failed).")
                else:
                    final_error_results.append("Error: Could not generate transcription for batch.")
            return final_error_results

    def asr(self, audio_bytes: bytes) -> str:
        logger.debug("Single instance ASR called (will use batch processing internally).")
        results = self.asr_batch([audio_bytes])
        return results[0] if results else "Error: ASR processing failed."

# __main__ block (as in previous version, ensure logging.getLogger().setLevel(logging.DEBUG) is active here for testing)
if __name__ == '__main__':
    # Ensure the root logger is set to DEBUG for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Alternatively, if basicConfig was already called by a library, just set the level for the root or specific loggers:
    # logging.getLogger().setLevel(logging.DEBUG) 
    # logging.getLogger(__name__).setLevel(logging.DEBUG) # For this module's logger
    # logging.getLogger("src.asr_manager").setLevel(logging.DEBUG) # If using named loggers like logger = logging.getLogger(__name__)

    print("Running ASRManager standalone test with BATCHING (DEBUG logging enabled)...")
    # ... (rest of the __main__ block is the same as the one from the previous version with varied audio lengths)
    base_path_for_models = os.path.join(SCRIPT_DIR_ASR_MANAGER, "..", "..", "models") 
    model_name_for_testing = "asr_finetuned_model_small_aug"
    constructed_model_path_attempt1 = os.path.join(SCRIPT_DIR_ASR_MANAGER, "..", "..", "models", model_name_for_testing, "model_and_processor_files")
    constructed_model_path_attempt2 = os.path.join(SCRIPT_DIR_ASR_MANAGER, "..", "models", model_name_for_testing, "model_and_processor_files")

    if os.path.isdir(constructed_model_path_attempt1):
        constructed_model_path = constructed_model_path_attempt1
    elif os.path.isdir(constructed_model_path_attempt2):
        constructed_model_path = constructed_model_path_attempt2
    else: 
        constructed_model_path = f"../models/{model_name_for_testing}/model_and_processor_files" 
        if not os.path.isdir(constructed_model_path):
             constructed_model_path = f"../../models/{model_name_for_testing}/model_and_processor_files"

    print(f"Attempting to load model for testing from: {constructed_model_path}")

    if not os.path.isdir(constructed_model_path):
        print(f"ERROR: Model directory not found: {constructed_model_path}. Please check path for standalone test.")
    else:
        try:
            manager = ASRManager(model_path_override=constructed_model_path)
            print("ASRManager initialized.")

            dummy_sr = 16000
            audio_short = np.random.randn(dummy_sr * 1).astype(np.float32) 
            audio_medium = np.random.randn(dummy_sr * 15).astype(np.float32) 
            audio_long_exact = np.random.randn(dummy_sr * 30).astype(np.float32) 
            audio_very_long = np.random.randn(dummy_sr * 35).astype(np.float32)


            io_short = io.BytesIO(); sf.write(io_short, audio_short, dummy_sr, format='WAV', subtype='FLOAT'); bytes_short = io_short.getvalue()
            io_medium = io.BytesIO(); sf.write(io_medium, audio_medium, dummy_sr, format='WAV', subtype='FLOAT'); bytes_medium = io_medium.getvalue()
            io_long_exact = io.BytesIO(); sf.write(io_long_exact, audio_long_exact, dummy_sr, format='WAV', subtype='FLOAT'); bytes_long_exact = io_long_exact.getvalue()
            io_very_long = io.BytesIO(); sf.write(io_very_long, audio_very_long, dummy_sr, format='WAV', subtype='FLOAT'); bytes_very_long = io_very_long.getvalue()
            
            faulty_bytes = b"this is not valid audio data and will fail sf.read"
            # Create truly empty audio that soundfile can write (0 samples)
            empty_sound_data = np.array([], dtype=np.float32)
            io_empty = io.BytesIO(); sf.write(io_empty, empty_sound_data, dummy_sr, format='WAV', subtype='FLOAT'); bytes_empty = io_empty.getvalue()


            audio_batch = [bytes_short, faulty_bytes, bytes_medium, bytes_empty, bytes_long_exact, bytes_very_long]
            
            print(f"\nTesting ASR with a batch of {len(audio_batch)} audios of varying lengths...")
            logger.debug(f"Test batch audio byte lengths: {[len(b) for b in audio_batch]}")
            transcriptions = manager.asr_batch(audio_batch)
            
            print("\nBatch Transcriptions:")
            if len(transcriptions) == len(audio_batch):
                for i, t in enumerate(transcriptions): 
                    print(f"Audio {i+1} (Input audio_bytes length: {len(audio_batch[i]) if audio_batch[i] else 0} bytes): '{t}'")
            else:
                print(f"Error: Number of transcriptions ({len(transcriptions)}) does not match batch size ({len(audio_batch)}).")
                print("Transcriptions received:", transcriptions)
            
            print("\nTesting single ASR (via batch wrapper) with 1s audio:")
            single_transcription_short = manager.asr(bytes_short)
            print(f"Single 1s audio transcription: '{single_transcription_short}'")

            print("\nTesting single ASR (via batch wrapper) with 35s audio (should be truncated):")
            single_transcription_very_long = manager.asr(bytes_very_long)
            print(f"Single 35s audio transcription: '{single_transcription_very_long}'")

        except Exception as e:
            print(f"An error occurred during standalone testing: {e}")
            import traceback
            traceback.print_exc()