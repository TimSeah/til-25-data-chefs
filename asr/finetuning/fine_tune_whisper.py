import os
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorSpeechSeq2SeqWithPadding,
)
import evaluate
import numpy as np

# --- Configuration ---
# Dataset paths
DATASET_PATH = "/home/jupyter/advanced/asr"  # Directory containing asr.jsonl and audio files
JSON_MANIFEST_FILE = os.path.join(DATASET_PATH, "asr.jsonl")

# Model configuration
BASE_MODEL_ID = "openai/whisper-small"  # You can choose "openai/whisper-tiny", "openai/whisper-base", etc.
LANGUAGE = "english" # Set to your dataset's language
TASK = "transcribe"    # Or "translate" if that's your goal

# Output directory for the fine-tuned model
OUTPUT_MODEL_DIR = "/home/jupyter/til-25-data-chefs/models/my_whisper_finetuned"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Training arguments (adjust as needed)
TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_MODEL_DIR, "training_checkpoints"), # Checkpoints saved here
    per_device_train_batch_size=8,  # Reduce if OOM, increase if GPU memory allows
    gradient_accumulation_steps=2,  # Effective batch size = per_device * num_devices * accumulation
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000, # Or num_train_epochs
    # num_train_epochs=3, # Alternative to max_steps
    gradient_checkpointing=True, # Saves memory
    fp16=torch.cuda.is_available(), # Use mixed precision if a CUDA GPU is available
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
    remove_unused_columns=False, # Important for custom dataset loading
)

# --- 1. Load Feature Extractor, Tokenizer, and Processor ---
print(f"Loading feature extractor, tokenizer, and processor for {BASE_MODEL_ID}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language=LANGUAGE, task=TASK)

# --- 2. Load and Prepare Dataset ---
print(f"Loading dataset from {JSON_MANIFEST_FILE}...")
# The 'audio' column in your jsonl points to filenames.
# We need to make sure `load_dataset` knows where to find these audio files.
# The paths in 'asr.jsonl' are relative to DATASET_PATH.
raw_dataset = load_dataset("json", data_files={"train": JSON_MANIFEST_FILE}, field="data")["train"] # Assuming 'data' is not a field and all lines are items

# If your asr.jsonl is just a list of json objects per line, it's simpler:
# raw_dataset = load_dataset("json", data_files={"train": JSON_MANIFEST_FILE})["train"]


# Add absolute path to audio files
def resolve_audio_path(batch):
    batch["audio_path"] = os.path.join(DATASET_PATH, batch["audio"])
    return batch

raw_dataset = raw_dataset.map(resolve_audio_path)

# Cast the 'audio_path' to Audio feature to load and resample
raw_dataset = raw_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
print(f"First example: {raw_dataset[0]}")


# Split dataset (e.g., 90% train, 10% test/validation)
# Ensure you have enough data for a meaningful split. If not, you might use a small fixed validation set.
if len(raw_dataset) > 100: # Arbitrary threshold for splitting
    train_test_split = raw_dataset.train_test_split(test_size=0.1)
    dataset_splits = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
else: # If dataset is too small, use it all for training and potentially evaluate on a separate small set later
    print("Dataset is small, using all for training. Consider a separate validation set for robust evaluation.")
    dataset_splits = DatasetDict({"train": raw_dataset})


# --- 3. Preprocessing Function ---
def prepare_dataset(batch):
    # Load and resample audio data (handled by .cast_column to Audio)
    audio = batch["audio_path"] # This now contains the loaded and resampled audio data

    # Compute log-Mel input features
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

print("Preprocessing dataset...")
tokenized_datasets = dataset_splits.map(prepare_dataset, num_proc=1) # Adjust num_proc based on your CPU cores
print("Dataset preprocessing complete.")

# --- 4. Data Collator ---
# This will dynamically pad the features and labels
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Evaluation Metric (WER) ---
print("Setting up evaluation metric (WER)...")
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 6. Load Pre-trained Model ---
print(f"Loading base model {BASE_MODEL_ID} for fine-tuning...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

# Configure model for training
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
if hasattr(model.config, "use_cache"): # Important for gradient checkpointing
    model.config.use_cache = False


# --- 7. Initialize Trainer ---
print("Initializing Seq2SeqTrainer...")
trainer = Seq2SeqTrainer(
    args=TRAINING_ARGS,
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("test"), # Use .get() in case 'test' split doesn't exist
    data_collator=data_collator,
    compute_metrics=compute_metrics if tokenized_datasets.get("test") else None,
    tokenizer=processor.feature_extractor, # Pass feature_extractor for generation during evaluation
)

# --- 8. Start Fine-Tuning ---
print("Starting fine-tuning...")
if tokenized_datasets.get("train"):
    train_result = trainer.train()
    print("Fine-tuning complete.")

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
else:
    print("No training data found after splitting. Skipping training.")


# --- 9. Save Fine-Tuned Model & Processor ---
print(f"Saving fine-tuned model and processor to {OUTPUT_MODEL_DIR}...")
trainer.save_model(OUTPUT_MODEL_DIR) # Saves model config and weights
processor.save_pretrained(OUTPUT_MODEL_DIR) # Saves processor (feature_extractor & tokenizer)
print("Model and processor saved successfully.")

if tokenized_datasets.get("test"):
    print("Evaluating final model on the test set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
else:
    print("No test/evaluation data found after splitting. Skipping final evaluation.")

print(f"Fine-tuning script finished. Model saved in {OUTPUT_MODEL_DIR}")