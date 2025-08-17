from datasets import load_dataset


raw_data = load_dataset("glue","mrpc")
raw_train_data = raw_data["train"]
# print(raw_data)

print(raw_train_data)
print(raw_train_data.features)
print(raw_train_data["sentence1"])


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# this only return the list of list
# tokenized_dataset = tokenizer(
#     raw_train_data["sentence1"][:2],
#     raw_train_data["sentence2"][:2],
#     padding=True,
#     truncation=True,
# )

# print(tokenized_dataset)


print(tokenizer("helel the bes if "), padding=True, truncation=True)