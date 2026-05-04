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
    
x=torch.rand(2, 3, 768)
F=FeedForward(x)
print(F.shape)