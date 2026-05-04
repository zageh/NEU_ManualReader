import torch
import torch.nn as nn

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