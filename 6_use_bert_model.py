from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model_name = "D:/A_Code/model/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = classifier("我今天很开心")
print(result)








