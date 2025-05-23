import os
import json # Kept import in case you want to use it later, though not used in this version
from collections import Counter

def create_paddle_artifacts(dataset_base_dir, output_src_dir):
    """
    Prepares dataset artifacts for PaddleOCR training.
    Ensures multi-line transcriptions are converted to single lines.

    Args:
        dataset_base_dir (str): The root directory of the raw dataset (e.g., '~/advanced/ocr').
        output_src_dir (str): The directory where 'train_label.txt' and
                              'custom_char_dict.txt' will be saved (e.g., '~/til-25-data-chefs/ocr/src').
    """
    print(f"Starting dataset preparation...")
    print(f"Reading from: {os.path.abspath(dataset_base_dir)}")
    print(f"Will write artifacts to: {os.path.abspath(output_src_dir)}")

    os.makedirs(output_src_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp') # Added more common extensions
    lines_for_label_file = []
    all_transcriptions_for_dict = []

    # --- Create Label File ---
    processed_images = 0
    skipped_images_no_txt = 0
    skipped_images_empty_txt = 0

    print(f"Scanning for image files in {os.path.abspath(dataset_base_dir)}...")
    for filename in os.listdir(dataset_base_dir):
        if filename.lower().endswith(image_extensions):
            image_basename_with_ext = filename
            image_basename_no_ext, _ = os.path.splitext(filename)

            # Path for the label file should be relative to dataset_base_dir
            # as this dir will be specified as data_dir in PaddleOCR config
            image_path_for_label = image_basename_with_ext

            txt_filename = image_basename_no_ext + "_text.txt"
            txt_filepath = os.path.join(dataset_base_dir, txt_filename)

            if os.path.exists(txt_filepath):
                try:
                    with open(txt_filepath, 'r', encoding='utf-8') as f:
                        # Read all lines, strip each line, then join with spaces
                        raw_lines = f.readlines()
                        # Process to handle newlines correctly for single-line label
                        transcription_parts = [line.strip() for line in raw_lines]
                        transcription = " ".join(transcription_parts)
                        # Normalize multiple spaces that might result from empty lines or excessive spacing
                        transcription = " ".join(transcription.split())


                    if transcription: # Check if not empty after processing
                        lines_for_label_file.append(f"{image_path_for_label}\t{transcription}")
                        all_transcriptions_for_dict.append(transcription)
                        processed_images += 1
                    else:
                        print(f"Warning: Empty transcription in {txt_filename} for {image_basename_with_ext} after processing.")
                        skipped_images_empty_txt += 1
                except Exception as e:
                    print(f"Error reading or processing {txt_filepath}: {e}")
            else:
                # This case should be less common if ocr.jsonl was accurate and all _text.txt files exist
                print(f"Warning: No corresponding _text.txt file found for {image_basename_with_ext} (looked for {txt_filename})")
                skipped_images_no_txt += 1

    print(f"\nLabel file processing summary:")
    print(f"  Processed images with text: {processed_images}")
    print(f"  Skipped (no _text.txt file): {skipped_images_no_txt}")
    print(f"  Skipped (empty _text.txt file after processing): {skipped_images_empty_txt}")

    if not lines_for_label_file:
        print("\nCRITICAL: No image-transcription pairs found. Cannot create label file.")
        # If you want the script to exit here, uncomment the next line
        # return
    else:
        output_label_filepath = os.path.join(output_src_dir, "train_label.txt")
        with open(output_label_filepath, 'w', encoding='utf-8') as f:
            for line in lines_for_label_file:
                f.write(line + "\n")
        print(f"\nSuccessfully created label file: {output_label_filepath} with {len(lines_for_label_file)} entries.")

    # --- Create Character Dictionary ---
    if not all_transcriptions_for_dict:
        print("CRITICAL: No transcriptions collected. Cannot create character dictionary.")
        return # Exit if no text to build dict from

    char_counts = Counter()
    for text in all_transcriptions_for_dict:
        for char in text:
            char_counts[char] += 1

    # Standard practice: <blank> token is often at index 0, but PaddleOCR might handle it differently
    # or expect it to be part of the character_list if used.
    # For PP-OCR models, usually, the dictionary does not explicitly include a <blank> token
    # as CTC loss handles blank internally. Let's build from data only.
    # If your specific model/config requires '<blank>', you might need to add it.
    # character_list = ['<blank>'] # Original line, let's reconsider for typical PaddleOCR usage

    # Create char list from data, ensure ' ' (space) is included if present.
    # Sorting ensures consistent dictionary order.
    unique_chars_from_data = sorted(list(char_counts.keys()))
    character_list = []
    
    # Add ' ' space character first if it's in the data, as it's fundamental.
    # Some models/configs might have specific expectations for space.
    if ' ' in unique_chars_from_data:
        character_list.append(' ') # Ensure space is included
        unique_chars_from_data.remove(' ')
    
    character_list.extend(unique_chars_from_data) # Add remaining sorted characters

    # For many PaddleOCR setups, the first character in the dict is treated as blank by CTC if not specified otherwise.
    # Often, people add a dummy/unused character at the beginning if their actual data doesn't have a natural "blank"
    # or if they want to explicitly control it. For simplicity, let's use what's in the data.
    # If you encounter issues, consult your specific PaddleOCR model's documentation on char_dict.txt format.
    # A common practice for PP-OCR is to have `unk` character for unknown characters, often represented by space.
    # The default en_dict.txt starts with ' ' (space).

    output_dict_filepath = os.path.join(output_src_dir, "custom_char_dict.txt")
    with open(output_dict_filepath, 'w', encoding='utf-8') as f:
        for char in character_list:
            f.write(char + "\n")
    print(f"Successfully created character dictionary: {output_dict_filepath} with {len(character_list)} characters.")
    print(f"Top 10 most common characters (after potential modifications): {char_counts.most_common(10)}")
    print(f"Character dictionary starts with: {character_list[:10]}") # Show more initial chars

    print("\nDataset preparation finished.")
    print(f"Please ensure your PaddleOCR training configuration points to:")
    print(f"  Label file: {os.path.abspath(output_label_filepath)}")
    print(f"  Character dict: {os.path.abspath(output_dict_filepath)}")
    print(f"  And an image_root_dir (data_dir in YAML) like: {os.path.abspath(dataset_base_dir)}")


if __name__ == "__main__":
    # These paths will be expanded to absolute paths
    raw_dataset_dir = os.path.expanduser("~/advanced/ocr")
    destination_src_dir = os.path.expanduser("~/til-25-data-chefs/ocr/src")

    # Basic validation of paths
    if not os.path.isdir(raw_dataset_dir):
        print(f"ERROR: Raw dataset directory not found: {raw_dataset_dir}")
        print("Please ensure the directory exists and contains your images and _text.txt files.")
    elif not os.path.exists(os.path.dirname(destination_src_dir)):
        # Check if the parent of the destination exists, to avoid errors if destination_src_dir itself
        # is a deeper path that needs intermediate directories created by os.makedirs.
        # os.makedirs will create destination_src_dir, but its parent must be valid.
        print(f"ERROR: Parent directory for destination output not found: {os.path.dirname(destination_src_dir)}")
        print(f"Please ensure the path {os.path.dirname(destination_src_dir)} is valid.")
    else:
        create_paddle_artifacts(dataset_base_dir=raw_dataset_dir,
                                output_src_dir=destination_src_dir)