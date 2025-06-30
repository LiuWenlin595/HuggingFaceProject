from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "uer/gpt2-chinese-cluecorpussmall"
model_name = "Qwen/Qwen2-0.5B-Instruct"
# cache_dir = "D:/A_Code/model/uer/gpt2-chinese-cluecorpussmall"
cache_dir = "D:/A_Code/model/Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型加载完成，缓存路径为：{cache_dir}")


