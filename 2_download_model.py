from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "uer/gpt2-chinese-cluecorpussmall"
model_name = "google-bert/bert-base-chinese"
# cache_dir = "D:/A_Code/model/uer/gpt2-chinese-cluecorpussmall"
cache_dir = "D:/A_Code/model/google-bert/bert-base-chinese"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型加载完成，缓存路径为：{cache_dir}")


