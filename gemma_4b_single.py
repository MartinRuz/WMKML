from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-4b-it"

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto",
    torch_dtype=torch.float16
).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Was kannst du mir über Friedrich Merz sagen?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)