import tiktoken

import torch
from torch.utils.data import Dataset,DataLoader

with open(r"project1\src\data\output.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
eos_token="<|endoftext|>"

class gpt_dataset_v1(Dataset):
    def __init__(self,txt,tokenizer,max_len,stride):
        self.target_ids=[]
        self.input_ids=[]
        
        token_ids=tokenizer.encode(txt,allowed_special={eos_token})
        
        for i in range(1,len(token_ids)-max_len,stride):
            input_txt=token_ids[i:i+max_len]
            target_txt=token_ids[i+1:max_len+i+1]
            self.target_ids.append(torch.tensor(target_txt))
            self.input_ids.append(torch.tensor(input_txt))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    
def create_dataloader_v1(txt,batch_size=4,max_len=256,stride=128,
                      shuffle=True,drop_last=True,
                      num_workers=0):
    
    tokenizer=tiktoken.get_encoding("gpt2")
    
    dataset=gpt_dataset_v1(txt,tokenizer,max_len,stride)
    
    dataloader=DataLoader(dataset,batch_size=batch_size,
                          shuffle=shuffle,drop_last=drop_last,
                          num_workers=num_workers)
    
    return dataloader

dataloader=create_dataloader_v1(raw_text,batch_size=8,max_len=128,
                                stride=4,shuffle=False)

data_iter=iter(dataloader)
inputs,targets=next(data_iter)

vocab_size=50257
embed_dim=768

embedding_layer=torch.nn.Embedding(vocab_size,embed_dim)

print(embedding_layer(inputs).shape)