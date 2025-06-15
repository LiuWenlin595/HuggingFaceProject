import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = ""  # hf_liuwenlin_QjcEuBGysfuluLzTPpWGfWuDJiYoemABtU
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(API_URL, headers=headers, json={"inputs": "你好"})

print(response)


