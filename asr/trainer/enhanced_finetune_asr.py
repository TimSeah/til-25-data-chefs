import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
import audiomentations as AA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Core Configuration ---
# !!! ============== IMPORTANT: DEFINE YOUR EXPERIMENT NAME FOR EACH RUN ============== !!!
EXPERIMENT_NAME = "asr_finetuned_v3"  # E.g., "run_with_more_noise_10_epochs"
# !!! ================================================================================ !!!

# ... (other configs) ...

# --- Paths ---
DATASET_PATH = os.path.expanduser("~/advanced/asr/asr.jsonl") # Ensure this path is correct
DATASET_DIR = os.path.dirname(DATASET_PATH)

MODELS_BASE_DIR = os.path.join(SCRIPT_DIR, "..", "models")
LOGS_BASE_DIR = os.path.join(SCRIPT_DIR, "..", "logs")

# >>> CHANGE THIS LINE <<<
OUTPUT_DIR_NAME = EXPERIMENT_NAME # Use the EXPERIMENT_NAME for the output directory name
MODEL_PROCESSOR_SUBDIR = "model_and_processor_files"

OUTPUT_DIR = os.path.join(MODELS_BASE_DIR, OUTPUT_DIR_NAME)
LOGGING_DIR = os.path.join(LOGS_BASE_DIR, f"{OUTPUT_DIR_NAME}_logs") # Logs will also use this name

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# --- Sanity Checks ---
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset JSONL file not found at {DATASET_PATH}.")
if AUGMENTATION_ENABLED and ("AddBackgroundNoise" in str(AA.Compose) and not (os.path.exists(PATH_TO_NOISE_FILES) and os.listdir(PATH_TO_NOISE_FILES))):
    print(f"WARNING: For 'AddBackgroundNoise' augmentation, '{PATH_TO_NOISE_FILES}' is not found or is empty. "
          "This augmentation might be ineffective or skipped by audiomentations if it can't find sounds.")

print(f"--- Experiment: {EXPERIMENT_NAME} ---")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Logging directory: {LOGGING_DIR}")
print(f"Number of epochs: {NUM_TRAIN_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}, Scheduler: {LR_SCHEDULER_TYPE}")
print(f"Data augmentation enabled: {AUGMENTATION_ENABLED}")


# --- Dataset Loading and Preparation ---
print(f"Loading dataset from: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def make_audio_path_absolute(example):
    audio_info = example.get("audio")
    if audio_info is None: return example
    path_to_check, is_dict_format = (audio_info["path"], True) if isinstance(audio_info, dict) and "path" in audio_info else (audio_info, False) if isinstance(audio_info, str) else (None, False)
    if path_to_check and not os.path.isabs(path_to_check):
        abs_path = os.path.join(DATASET_DIR, path_to_check)
        if is_dict_format: example["audio"]["path"] = abs_path
        else: example["audio"] = abs_path
    return example

print(f"Resolving audio file paths relative to: {DATASET_DIR}")
dataset = dataset.map(make_audio_path_absolute, num_proc=1) # This is usually fast
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# --- Initialize Processor, Tokenizer, Feature Extractor ---
print(f"Initializing components for model: {BASE_MODEL_ID}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)

# --- Load Model (with Flash Attention 2 if available) ---
print(f"Loading base model: {BASE_MODEL_ID}...")
try:
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        attn_implementation="flash_attention_2"
    )
    print("Successfully loaded model with Flash Attention 2.")
except ImportError:
    print("Flash Attention 2 not available (flash-attn package not installed or incompatible). Falling back.")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
except Exception as e:
    print(f"Could not load model with Flash Attention 2 (Error: {e}). Falling back.")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

TOKENIZER_MAX_LENGTH = model.generation_config.max_length if hasattr(model.generation_config, 'max_length') else 448
print(f"Using tokenizer max_length: {TOKENIZER_MAX_LENGTH}")

# --- Prepare Dataset Functions ---
def prepare_train_dataset(batch, tokenizer_obj, max_len):
    batch["labels"] = tokenizer_obj(batch["transcript"], truncation=True, max_length=max_len -1).input_ids
    return batch

def prepare_test_dataset(batch, feature_extractor_obj, tokenizer_obj, max_len):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor_obj(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer_obj(batch["transcript"], truncation=True, max_length=max_len -1).input_ids
    return batch

# --- Split and Map Dataset ---
print("Splitting dataset...")
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Configure lang_to_id and task_to_id in model.generation_config
if not hasattr(model.generation_config, "lang_to_id") or model.generation_config.lang_to_id is None:
    lang_code = "en" if LANGUAGE.lower() == "english" else None
    if lang_code: model.generation_config.lang_to_id = {f"<|{lang_code}|>": tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")}
if not hasattr(model.generation_config, "task_to_id") or model.generation_config.task_to_id is None:
    model.generation_config.task_to_id = {TASK: tokenizer.convert_tokens_to_ids(f"<|{TASK}|>")}
model.generation_config.language = LANGUAGE.lower()
model.generation_config.task = TASK

num_available_cores = os.cpu_count()
num_mappers = num_available_cores // 2 if num_available_cores and num_available_cores > 1 else 1
print(f"Using {num_mappers} processes for dataset mapping.")

print("Mapping datasets...")
train_dataset = train_dataset.map(
    prepare_train_dataset,
    fn_kwargs={"tokenizer_obj": tokenizer, "max_len": TOKENIZER_MAX_LENGTH},
    num_proc=num_mappers
)
test_dataset = test_dataset.map(
    prepare_test_dataset,
    fn_kwargs={"feature_extractor_obj": feature_extractor, "tokenizer_obj": tokenizer, "max_len": TOKENIZER_MAX_LENGTH},
    num_proc=num_mappers,
    remove_columns=dataset.column_names
)

# --- Augmentation Pipeline ---
augment_pipeline = None
if AUGMENTATION_ENABLED:
    print("Configuring audio augmentation pipeline...")
    augment_transforms = [
        AA.AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.0075, p=0.4), # Reduced max amplitude
        AA.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3), # Slightly wider stretch
        AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.3), # Slightly wider pitch
        # AA.Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.2), # Optional: Gain
    ]
    if os.path.exists(PATH_TO_NOISE_FILES) and os.listdir(PATH_TO_NOISE_FILES):
        print(f"Adding background noise from: {PATH_TO_NOISE_FILES}")
        augment_transforms.append(
            AA.AddBackgroundNoise(sounds_path=PATH_TO_NOISE_FILES, min_snr_in_db=3.0, max_sn_r_in_db=20.0, p=0.4)
        )
    else:
        print(f"Skipping AddBackgroundNoise as noise directory ('{PATH_TO_NOISE_FILES}') is not valid or empty.")
    augment_pipeline = AA.Compose(augment_transforms)

# --- Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndAugmentation:
    processor: WhisperProcessor
    decoder_start_token_id: int
    augmenter: AA.Compose = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        processed_input_features_list = []
        if "input_features" not in features[0]: # Needs feature extraction (typically train set)
            for feature in features:
                audio_dict = feature["audio"]
                raw_audio_array = audio_dict["array"].astype(np.float32) # Ensure float32
                sampling_rate = audio_dict["sampling_rate"]
                if self.augmenter:
                    augmented_audio = self.augmenter(samples=raw_audio_array, sample_rate=sampling_rate)
                else:
                    augmented_audio = raw_audio_array
                inputs = self.processor.feature_extractor(augmented_audio, sampling_rate=sampling_rate, return_tensors="pt")
                processed_input_features_list.append({"input_features": inputs.input_features[0]})
        else: # input_features already precomputed (typically test/eval set)
            processed_input_features_list = [{"input_features": f["input_features"]} for f in features]

        batch = self.processor.feature_extractor.pad(processed_input_features_list, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndAugmentation(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    augmenter=augment_pipeline
)

# --- Metrics ---
print("Loading WER metric...")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
    wer = 1.0 if not label_str and not pred_str else wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- Training Arguments ---
# Determine number of dataloader workers
dataloader_num_workers = num_available_cores // 2 if num_available_cores and num_available_cores > 2 else 2 # Min 2 if possible
if dataloader_num_workers == 0 : dataloader_num_workers = 1 # Ensure at least 1
print(f"Using {dataloader_num_workers} dataloader workers.")

training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "training_checkpoints"),
    per_device_train_batch_size=4, # Adjust based on your GPU memory
    gradient_accumulation_steps=4, # Effective batch size = 16
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE, # << Cosine scheduler
    warmup_ratio=0.1, # Standard warmup
    num_train_epochs=NUM_TRAIN_EPOCHS,
    gradient_checkpointing=True, # Saves memory at a small cost to speed
    fp16=torch.cuda.is_available(), # Mixed precision if CUDA is available
    # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # Uncomment if on Ampere+ and want to try bf16
    torch_compile=True, # << PyTorch 2.0+ feature for speedup
    logging_dir=LOGGING_DIR,
    logging_strategy="steps",
    logging_steps=50, # Log more frequently
    per_device_eval_batch_size=4, # Adjust based on GPU memory
    eval_strategy="steps",
    eval_steps=500, # Evaluate every 500 steps (adjust based on dataset size)
    save_strategy="steps",
    save_steps=500, # Save checkpoint every 500 steps
    save_total_limit=3, # Keep best, last, and one more recent checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=TOKENIZER_MAX_LENGTH,
    dataloader_num_workers=dataloader_num_workers, # << For faster data loading
    report_to=["tensorboard"],
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# --- Start Training ---
print(f"Starting training for experiment: {EXPERIMENT_NAME}...")
if torch.cuda.is_available():
    print(f"CUDA Initial GPU memory: Allocated={torch.cuda.memory_allocated(0)/1024**2:.2f}MB, Reserved={torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
    torch.cuda.empty_cache()
    print(f"CUDA Post-cache-clear GPU memory: Allocated={torch.cuda.memory_allocated(0)/1024**2:.2f}MB, Reserved={torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

train_result = trainer.train()
print(f"Training completed for {EXPERIMENT_NAME}.")

# --- Save Model, Metrics, and State ---
trainer.save_model() # Saves the tokenizer too by default if tokenizer passed to Trainer
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

final_model_save_path = os.path.join(OUTPUT_DIR, MODEL_PROCESSOR_SUBDIR)
os.makedirs(final_model_save_path, exist_ok=True) # Should already exist from training_args.output_dir
print(f"Saving final best model and processor explicitly to: {final_model_save_path}")
model.save_pretrained(final_model_save_path) # Save the model
processor.save_pretrained(final_model_save_path) # Save the processor (feature_extractor and tokenizer)
print(f"Best model and processor saved to {final_model_save_path}")

# --- Final Evaluation ---
print(f"Evaluating the best model for {EXPERIMENT_NAME} on the test set...")
eval_metrics = trainer.evaluate(eval_dataset=test_dataset) # Ensure evaluation on the test_dataset
eval_metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
print(f"Final evaluation results for {EXPERIMENT_NAME}: {eval_metrics}")

print(f"--- Experiment {EXPERIMENT_NAME} Finished ---")
