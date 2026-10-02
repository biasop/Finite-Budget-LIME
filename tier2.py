import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from my_core import sample_masks

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentence = "This movie is really good."

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

encoded = tokenizer(
    sentence,
    return_tensors="pt",
    return_special_tokens_mask=True,
)

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
free_idx = torch.where(
    (encoded["special_tokens_mask"][0] == 0)
    & (encoded["attention_mask"][0] == 1)
)[0]


with torch.inference_mode():
    logits = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    ).logits
    target_class = int(logits.argmax(dim=-1).item())

print("Tất cả token:", tokens)
print("Vị trí được che:", free_idx.tolist())
print("d =", len(free_idx))
print("Class cố định:", target_class)