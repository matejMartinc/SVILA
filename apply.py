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