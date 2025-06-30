from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model_name = "D:/A_Code/model/Qwen/Qwen2-0.5B-Instruct/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"

pretrained_model = BertModel.from_pretrained(pretrained_model_name).to(DEVICE)

# print(pretrained_model)
# print(pretrained_model.embeddings.word_embeddings)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 上游任务不参与训练
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # 下游任务参与训练
        outputs = self.fc(outputs.last_hidden_state[:, 0])
        outputs = outputs.softmax(dim=1)
        return outputs

model = Model().to(DEVICE)











