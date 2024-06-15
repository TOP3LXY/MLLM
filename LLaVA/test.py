import torch
from PIL import Image
from transformers import(
    LlavaForConditionalGeneration,
    LlavaProcessor
)

LLaVA_dir = "ckpt/model_001"

model_processor = LlavaProcessor.from_pretrained(LLaVA_dir)
model = LlavaForConditionalGeneration.from_pretrained(LLaVA_dir, device_map='cuda:0', torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<image>\nWhat's the content of the image?"},
]
prompt = model_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
img_dir = "test.jpg"
image = Image.open(img_dir)
inputs = model_processor(text=prompt, images=image, return_tensors="pt")

inputs = inputs.to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=20)
response = model_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
