import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaConfig,
)

qwen_tokenizer_dir = "ckpt/Qwen1.5-4B-Chat"
clip_dir = "ckpt/clip-vit-large-patch14-336"

# load models and tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_tokenizer_dir, device_map="cuda:0", torch_dtype=torch.bfloat16
)
clip_model = AutoModel.from_pretrained(
    clip_dir, device_map="cuda:0", torch_dtype=torch.bfloat16
)

# initial LLaVA
vision_config = clip_model.vision_model.config
llm_config = qwen_model.config
llava_config = LlavaConfig(vision_config=vision_config, text_config=llm_config)
model = LlavaForConditionalGeneration(llava_config)

# replace weight
model.vision_tower.vision_model = clip_model.vision_model
model.language_model = qwen_model

model.config.pad_token_id = qwen_tokenizer.pad_token_id
model.config.image_token_index = qwen_tokenizer.encode("<image>")[0]

model.save_pretrained("ckpt/model_001")
qwen_tokenizer.save_pretrained("ckpt/model_001")

autoprocesser = AutoProcessor.from_pretrained(clip_dir)
autoprocesser.save_pretrained("ckpt/model_001p")

# 需要把show_model/model_001p里面的preprocessor_config.json文件，放在show_model/model_001里面
