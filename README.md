---
license: mit
language:
- en
base_model:
- google/gemma-3-4b-it
pipeline_tag: text-generation
tags:
- Vision language model (VLM)
- Slovenian
---
# SVILA - Slovenian Vision Language Assistant #

## The models are based on google/gemma-3 series and were fine-tuned on curated instruction-tuning text-image Slovenian dataset using a custom SFT trainer. 

### The models are available on Hugging Face: 
- https://huggingface.co/GaMS-Beta/SVILA-1-12B
- https://huggingface.co/GaMS-Beta/SVILA-1-4B

## Dataset with instructions on how to compile it is available here:
- https://clarin.si/repository/xmlui/handle/11356/2050


## How to train the model: ##

### We offer the script to train the model on the EuroHPC cluster Leonardo Booster. 

- In the leonardo_finetune_ds_gemma.sh, change the '#SBATCH --account=ACCOUNT_NAME' to your account, change the path to your virtualenv and the path to your .json datafile

- Default backbone in the script is Gemma-3-4b-it, you can change the default model in the leonardo_finetune_ds_gemma.sh script (also change the batch size of 8 to 4 in that case)

```
# Create virtual env
python3 -m venv . /leonardo_work/ACCOUNT_NAME/venv
# install requirements
pip install -r requirements.txt
# start the fine-tuning
sbatch leonardo_finetune_ds_gemma.sh
```

## How to run it: ##

```python
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch


model_id = "output/SVILA-1-4B"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": ""}]
    },
    {
        "role": "user",
        "content": [
           {"type": "image", "image": "https://www.dangerous-business.com/wp-content/uploads/2024/02/DSC02109.jpg"}, 
           {"type": "text", "text": "Kaj je na sliki?"}
        ]
    }
]

print(processor.apply_chat_template(messages, tokenize=False))

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)


input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=500)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
```
