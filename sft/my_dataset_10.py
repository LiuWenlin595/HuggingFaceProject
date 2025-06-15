from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk(dataset_path="D:/A_Code/datasets/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]
        return text, label
    
if __name__ == "__main__":
    dataset = MyDataset(split="train")
    print(len(dataset))
    for data in dataset:
        print(data)





