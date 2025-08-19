from datasets import load_dataset


raw_data = load_dataset("glue","mrpc")
raw_train_data = raw_data["train"]
# print(raw_data)

print(raw_train_data)
print(raw_train_data.features)
print(raw_train_data["sentence1"])


