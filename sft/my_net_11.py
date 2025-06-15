from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model_name = "D:/A_Code/model/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

pretrained_model = BertModel.from_pretrained(pretrained_model_name).to(DEVICE)

# print(pretrained_model)
print(pretrained_model.embeddings.word_embeddings)

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











