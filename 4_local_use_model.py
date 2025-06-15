from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

model_dir = "D:/A_Code/model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# print(model)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

text = "你好，我是一款语言模型"

# result = pipe(text, max_length=10, num_return_sequences=1)
# result = pipe(text, max_new_tokens=50, 
#               num_return_sequences=3,
#               truncation=True,  # 是否主动截断输入文本来适应模型最大输入长度
#               temperature=0.3,  # 控制生成文本的多样性, temperature<1, 高值更大, 在softmax后会获得更高概率
#               top_k=50,  # 从概率最高的k个词里面选
#               top_p=0.9,  # 从概率最高的p%的词里面选
#               clean_up_tokenization_spaces=False)  # 控制生成文本的多样性

# print(result)

