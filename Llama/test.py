from modeling_llama import LlamaConfig, LlamaForSequenceClassification
import torch

model_config = LlamaConfig(num_labels=3, pad_token_id=237)
model = LlamaForSequenceClassification(model_config)
model.eval()
input_ids1 = torch.tensor([231, 435, 6523, 235, 236])
input_ids2 = torch.tensor([2332, 4332, 654, 2435, 236])
labels = torch.tensor([1, 1]).view(2, 1)
input_ids = torch.concat(
    tensors=[input_ids1.unsqueeze(dim=0), input_ids2.unsqueeze(dim=0)], dim=0
)
responce, loss = model(input_ids, labels)

print(input_ids, "\n", responce, "\n", loss)
