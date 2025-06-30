import torch
from my_dataset_10 import MyDataset
from my_net_11 import Model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

token = AutoTokenizer.from_pretrained("D:/A_Code/model/Qwen/Qwen2-0.5B-Instruct/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d")
# 0.1B可以跑, 0.5B就报错了, token也报错?

# 自定义函数, 对数据进行编码处理, 要在训练时编码, 而不是创建数据集时
def collate_fn(data):
    sentences = [item[0] for item in data]
    labels = [item[1] for item in data]
    # 对句子进行编码
    inputs = token.batch_encode_plus(
        batch_text_or_text_pairs=sentences, 
        padding="max_length", 
        truncation=True, 
        max_length=256, 
        return_tensors="pt",
        return_length=True)
    input_ids = inputs["input_ids"].to(DEVICE) # 为什么用键取, 没解释清
    attention_mask = inputs["attention_mask"].to(DEVICE)
    token_type_ids = inputs["token_type_ids"].to(DEVICE)
    labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    return input_ids, attention_mask, token_type_ids, labels
    

train_dataset = MyDataset(split="train")

# collate_fn有一堆问题
train_loader = DataLoader(train_dataset, 
                          batch_size=32, 
                          shuffle=True, 
                          drop_last=True)
                          # collate_fn=collate_fn)

if __name__ == "__main__":
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 数据也要和模型一样放进DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                out = outputs.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(f"epoch {epoch}, step {i}, loss {loss.item()}, acc {acc}")
        # 保存模型
        torch.save(model.state_dict(), f"./sft/model_{epoch}.pt")
        print(f"epoch {epoch} 训练完成")





