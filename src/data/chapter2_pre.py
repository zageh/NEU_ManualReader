import os
import requests

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "src/data/the-verdict.txt"

os.makedirs("src/data", exist_ok=True)

response = requests.get(url)
with open(file_path, "w", encoding="utf-8") as f:
    f.write(response.text)#写入verdict
    
print("成功")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()#打开verdict
    
    
import re
text="I kept my eyes on the white light."
result=re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)

class simp_tokenizer1:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
        
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    #把单词变为数字
    
    def decode(self,ids):
        text=" ".join(self.int_to_str[i] for i in ids)
        text= re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
    #把数字变成单词
    
with open("src/data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
