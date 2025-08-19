from transformers import pipeline, AutoTokenizer,AutoModel
from torch import nn


checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
# Sentiment analysis pipeline
# the pipeline function do three thing for us tokenization --> model--logits--labels
classification = pipeline("sentiment-analysis")
result = classification(sentences)
print(result)


# how the pipline  function actually work under the hood
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)


inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

print(inputs)
print(outputs.last_hidden_state.shape) # final representation of the sentence

print(model.config.id2label)
# now we have to add the head to the hidden state features 

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# this include a linear layer for the head of sequence classification task
model_head = AutoModelForSequenceClassification.from_pretrained(checkpoint) 
outputs = model_head(**inputs)
print(outputs)

print(outputs.logits.shape)

print(outputs.logits)

predictions = nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

labels = model.config.id2label

import torch

print(torch.argmax(predictions,dim=1))



# if we have three or more label we can add our own head using linear layer and then fine tuning it with our own dataset


# like the distilbert is funetunig base model bert on sst_2 dataset 
# the base mod


# how to decode our token back to the text


print(tokenizer.decode([  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]))



encoded_input = tokenizer(
    ["How are you? this sentence is truncation to max_lenght of 5", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    # max_length=5,
    return_tensors="pt", # return list by default 
)
print(encoded_input)

print(tokenizer.decode(encoded_input["input_ids"][0]))



