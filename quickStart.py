from transformers import pipeline

# 创建一个情感分析pipeline
classifier = pipeline("sentiment-analysis")

# 测试一些文本
texts = [
    "我很喜欢这个产品，质量非常好！",
    "这个服务太糟糕了，完全不值这个价格。",
    "今天天气不错，心情很好。"
]

# 进行情感分析
results = classifier(texts)

# 打印结果
for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"情感: {result['label']}")
    print(f"置信度: {result['score']:.4f}")
    print("-" * 50)
