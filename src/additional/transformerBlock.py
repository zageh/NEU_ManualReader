import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 +
                          torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)))
                          * (x + 0.044715 * torch.pow(x, 3)))
        
class FeedForward(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            GELU(),
            nn.Linear(4 * d_in, d_in),
        )
        
    def forward(self, x):
        return self.layers(x)
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.eps = 1e-5
        
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        #初始为0或者1，
        #但是模型会自己调整
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) 
        #correction = 0更常用，
        # 现在是有偏估计： 因为mean是由样本数据算出的会比真实均值更贴近数据，
        # 从而，方差的计算中(num - mean)**2.sum() / n会偏小
        # 所以，有偏估计把n改为(n-1)，使方差更贴近真实
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class MultiHeadAttention(nn.Module):
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
        self.dropout=nn.Dropout(dropout)
        
        self.register_buffer('mask',torch.tril(torch.ones(context_len,context_len)))
        
    def forward(self,x):
        b,num_token,d_in=x.shape
        
        keys=self.W_key(x)
        queries=self.W_query(x)
        values=self.W_value(x)
        
        keys=keys.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        queries=queries.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        values=values.view(b,num_token,self.num_heads,self.head_dim).transpose(1,2)
        
        attn_scores=queries @ keys.transpose(2,3)
        
        mask_bool=self.mask[:num_token,:num_token].bool()
        attn_scores.masked_fill_(~mask_bool[:num_token,:num_token], -float('inf'))
        
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        context_vec=(attn_weights @ values).transpose(1,2)
        
        context_vec= context_vec.contiguous().view(b,num_token,self.d_out)
        
        context_vec= self.out_proj(context_vec)
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self,d_in,context_len,heads_num,dropout,biased=False):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_in=d_in,
            d_out=d_in,
            context_len=context_len,
            heads_num=heads_num,
            dropout=dropout,
            biased=biased
        )
        
        self.ffn = FeedForward(d_in)
        self.ln1 = LayerNorm(d_in)
        self.ln2 = LayerNorm(d_in)
        self.drop_shortcut = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = x
        
        x=self.ln1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x+=shortcut
        
        shortcut=x
        
        x=self.ln1(x)
        x=self.ffn(x)
        x=self.drop_shortcut(x)
        x+=shortcut
        
        return x