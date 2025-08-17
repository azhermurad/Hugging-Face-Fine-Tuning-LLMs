import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset


checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


# dataset load 
raw_data = load_dataset("glue","mrpc")

raw_train_data = raw_data["train"]

print(raw_train_data.features)







