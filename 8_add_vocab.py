from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

vocab = tokenizer.get_vocab()

print(len(vocab))
print("阳" in vocab)
print("光" in vocab)
print("阳光" in vocab)

tokenizer.add_tokens(new_tokens=["阳光", "大地"])
tokenizer.add_special_tokens({"eos_token": "[EOS]"})

vocab = tokenizer.get_vocab()

print(len(vocab))
print("阳光" in vocab)
print("大地" in vocab)
# print(tokenizer)

result = tokenizer.encode(text="阳光照在大地上[EOS]", 
                          text_pair=None, 
                          truncation=True, 
                          padding="max_length", 
                          max_length=10, 
                          add_special_tokens=True,
                          return_tensors=None)

print(result)
print(tokenizer.decode(result))


