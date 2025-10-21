import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image
import re
import numpy as np
import random

from params_gemma import DataArguments, SYSTEM_MESSAGE, DEFAULT_END_TOKEN, DEFAULT_START_TOKEN, \
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(images, processor):

    content = []

    for img in images:
        content.append({"type": "image", "image": img})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    vision_infos = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")

    pixel_values = vision_infos["pixel_values"]

    return pixel_values


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str | list,
            processor: transformers.ProcessorMixin,
            data_args: DataArguments,
            padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames
        self.max_len = data_args.max_seq_len

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]

            is_video = False
            num_frames = None
            pixel_values = None

            processor = self.processor
            if "image" in sources:
                image_files = sources["image"]
                if isinstance(image_files, str):
                    image_files = [image_files]

                images = []

                for image_file in image_files:
                    images.append(Image.open(image_file).convert("RGB"))

                pixel_values = get_image_info(images, processor)
            item_id = sources['id']
            sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

            text = processor.apply_chat_template(
                sources, add_generation_prompt=False, tokenize=False
            )

            input_ids = processor.tokenizer(text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)

            original_len = len(input_ids)
            if original_len > self.max_len:
                print(
                    f"\n[WARNING] Truncating sequence for item ID '{item_id}'. "
                    f"Original length {original_len} > max_len {self.max_len}. "
                    f"Sequence will be truncated."
                )
                input_ids = input_ids[:self.max_len]

            input_ids = input_ids.to(torch.long)
            labels = input_ids.clone()

            # Mask image tokens
            image_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
            image_end_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["eoi_token"])
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["image_token"])

            pad_token_id = processor.tokenizer.pad_token_id
            bos_token_id = processor.tokenizer.bos_token_id

            # Mask tokens for not being used in the loss computation
            labels[labels == pad_token_id] = IGNORE_INDEX
            labels[labels == bos_token_id] = IGNORE_INDEX
            labels[labels == image_start_token_id] = IGNORE_INDEX
            labels[labels == image_end_token_id] = IGNORE_INDEX
            labels[labels == image_token_id] = IGNORE_INDEX
            labels = labels.to(torch.long)

            attention_mask = (input_ids > -1000000).to(torch.long)

            data_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if pixel_values is not None:
                array_ids = input_ids
                token_type_ids = np.zeros_like(input_ids)
                token_type_ids[array_ids == processor.image_token_id] = 1
                token_type_ids = torch.tensor(token_type_ids)

                data_dict["pixel_values"] = pixel_values
                data_dict["token_type_ids"] = token_type_ids
            return data_dict
        except Exception as e:
            print("Data fetch error", e)
            return self.__getitem__(random.randint(0, len(self)))



class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_token_type_ids = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_token_type_ids.append(example["token_type_ids"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        batch_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            token_type_ids = pad_sequence(batch_token_type_ids, padding_side='right', padding_value=0)
            batch_dict.update(pixel_values=pixel_values, token_type_ids=token_type_ids)
        else:
            dummy_pixel_values_shape = (1, 3, 896, 896)  # Shape returned by the Gemma processor
            dummy_pixel_values = torch.zeros(dummy_pixel_values_shape)
            dummy_token_type_ids = torch.zeros((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long)
            batch_dict.update(pixel_values=dummy_pixel_values, token_type_ids=dummy_token_type_ids)
        return batch_dict


def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string


def replace_image_tokens(input_string):
    pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
    replacement = "\n\n" + VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN * 256 + VISION_END_TOKEN + "\n\n"

    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False, num_frames=None):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:

        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)

        transformed_content = replace_image_tokens(conversation["value"])
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)