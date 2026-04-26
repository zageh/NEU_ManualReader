import re

class simp_tokenizer2:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
        
    def encode(self,text,add_endoftext):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        preprocessed=[
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        
        if add_endoftext:
            preprocessed.append("<|endoftext|>")
        
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    #把单词变为数字
    
    def decode(self,ids):
        if hasattr(ids,'tolist'):
            ids=ids.tolist()
        
        text=" ".join(self.int_to_str[i] for i in ids)
        text= re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
    #把数字变成单词
    
with open(r"project1\src\data\the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed=[item.strip() for item in preprocessed if item.strip()]

all_tokens=sorted(list(set(preprocessed))) #sort为了确保每次同一单词都对应同一数字，不影响实际功能
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab={s:i for i,s in enumerate(all_tokens)}

tokenizer=simp_tokenizer2(vocab)

import torch
from torch.utils.data import Dataset

class gpt_dataset_v1(Dataset):
    def __init__(self,txt,tokenizer,max_len,stride):
        self.target_ids=[]
        self.input_ids=[]
        
        token_ids=tokenizer.encode(txt,add_endoftext=True)
        
        for i in range(1,len(token_ids)-max_len,stride):
            input_txt=token_ids[i:i+max_len]
            target_txt=token_ids[i+1:max_len+i+1]
            self.target_ids.append(torch.tensor(target_txt))
            self.input_ids.append(torch.tensor(input_txt))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    
dataset=gpt_dataset_v1(raw_text,tokenizer,max_len=128,stride=1)
x,y=dataset[0]
print(f"Total tokens in text: {len(tokenizer.encode(raw_text, add_endoftext=True))}") #少于tiktoken，但是主要是unk太暴力了
print(f"x:{x} -> {tokenizer.decode(x)}")
print(f"y:{y} -> {tokenizer.decode(y)}")