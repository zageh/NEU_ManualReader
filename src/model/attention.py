import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import tiktoken

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

dataloader=create_dataloader_v1(raw_text,batch_size=4,max_len=128,
                                stride=4,shuffle=False)

data_iter=iter(dataloader)
inputs,targets=next(data_iter)

vocab_size=50257
embed_dim=768

embedding_layer=nn.Embedding(vocab_size,embed_dim)

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,d_in,d_out,context_len,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads==0, "请修改num_heads"
        
        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        
        self.W_query=nn.Linear(d_in,d_out,bias=False)
        self.W_key  =nn.Linear(d_in,d_out,bias=False)
        self.W_value=nn.Linear(d_in,d_out,bias=False)
        
        self.out_proj=nn.Linear(d_out,d_out)
        #最终线性层，融合多层信息
        
        self.dropout=nn.Dropout(dropout)
        #dropout 随机把weight改为0，提高鲁棒性
        
        self.register_buffer('mask',torch.tril(torch.ones(context_len,context_len)))
        
    def forward(self,x):
        b,num_token,d_in=x.shape
        
        keys=self.W_key(x)
        queries=self.W_query(x)
        values=self.W_value(x)
        
        keys=keys.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        queries=queries.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        values=values.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        #d_out被拆分成了 head_dim 和 num_heads
        #view的作用： 把d_out 维转为 num_heads*head_dim 维
        
        attn_scores=queries @ keys.transpose(2,3)
        
        mask_bool=self.mask[:num_token,:num_token].bool()
        attn_scores.masked_fill_(~mask_bool[:num_token,:num_token], -float('inf'))
        #masked_fill_ 本地操作，节约显存
        
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        #这里的sqrt（d_k)原自于 总体的标准差 sqrt(方差), 
        #这个次数可以改，也就是scale或者temperature，
        #可以有attn_scores @ learned_scale
        
        context_vec=(attn_weights @ values).transpose(1,2)
        
        context_vec= context_vec.contiguous().view(b,num_token,self.d_out)
        
        context_vec= self.out_proj(context_vec)
        return context_vec
    
mha=MultiHeadAttentionWrapper(768,768,128,0.1,12)
output=mha(embedding_layer(inputs))

print(output.shape)
print(sum(p.numel() for p in mha.parameters()))