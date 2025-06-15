from datasets import load_dataset, load_from_disk

# 从huggingface下载数据集到本地
# dataset = load_dataset(
#     path="lansinuote/ChnSentiCorp",
#     # split=["train", "test", "validation"],
#     # cache_dir="D:/A_Code/datasets"
# )
# dataset.save_to_disk("D:/A_Code/datasets")
# print(dataset)


# 本地使用数据集
dataset = load_from_disk(dataset_path="D:/A_Code/datasets/ChnSentiCorp")
print(dataset)



