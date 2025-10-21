import json
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import copy
import re

from params_gemma import DataArguments, SYSTEM_MESSAGE, DEFAULT_END_TOKEN, DEFAULT_START_TOKEN, \
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN

def replace_image_tokens(input_string):
    pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
    replacement = "\n\n" + VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN * 256 + VISION_END_TOKEN + "\n\n"

    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False, num_frames=None):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:

        transformed_content = replace_image_tokens(conversation["value"])
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

# --- Configuration ---
MODEL_ID = "google/gemma-3-12b-it"
INPUT_JSON = "all_data_gemma_format_balanced_gams_nemotron_ft_3000_max_len.json"
OUTPUT_JSON = "all_data_gemma_format_PREPROCESSED_CLEAN.json"
MAX_LEN = 3000

# --- Load Processor ---
print(f"Loading processor from {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- Load Data ---
print(f"Loading data from {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding='utf-8') as f:
    data = json.load(f)

print(f"Original dataset size: {len(data)}")

# --- Filter Data ---
valid_data = []
removed_count = 0
for i, item in enumerate(tqdm(data, desc="Filtering examples")):
    try:
        # 1. Check for essential keys
        if 'id' not in item or 'conversations' not in item:
            removed_count += 1
            continue

        # 2. Check image validity (if present)
        if "image" in item:
            image_files = item["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            # Try to open every image to ensure it's not corrupt
            for image_file in image_files:
                Image.open(image_file).convert("RGB")
        conversations = item['conversations']
        is_video = False
        num_frames = None
        conversations = copy.deepcopy(llava_to_openai(conversations, is_video=is_video, num_frames=num_frames))

        # 3. Check sequence length
        text = processor.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        if len(processor.tokenizer(text, add_special_tokens=False)['input_ids']) > MAX_LEN:
            removed_count += 1
            continue

        # If all checks pass, add it to the valid list
        valid_data.append(item)

    except Exception as e:
        # Any other error (bad image path, etc.) means we discard the sample
        print(f"Warning: Discarding item at index {i} due to error: {e}")
        removed_count += 1
        continue

print(f"\nFinished filtering. New dataset size: {len(valid_data)}")
print(f"Removed {removed_count} examples ({removed_count / len(data) * 100:.2f}%)")

# --- Save New Dataset ---
print(f"Saving filtered data to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)

print("Done.")