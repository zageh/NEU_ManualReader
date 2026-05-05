import torch
import torch.nn as nn
from .attention import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_len": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
}
#确实更清爽了

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                d_in=cfg["emb_dim"], 
                d_out=cfg["emb_dim"],
                context_len=cfg["context_len"],
                num_heads=cfg["n_heads"], 
                dropout=cfg["drop_rate"]
            ) for _ in range(cfg["n_layers"])]
            #懒得改之前那个了
        )
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        tok_embeds = self.tok_emb(in_idx)
        
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)
        #把 token 编号改为位置编号
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        x = self.trf_blocks(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    
def generate_text_simp(model, idx, max_new_tokens,context_size):
    #idx==(batch, n_tokens)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1,:]
        #只要(batch, vocab_size)
        
        probas = torch.softmax(logits, dim=-1)
        
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        #获取概率最大的idx
        
        idx = torch.cat((idx, idx_next), dim=1)
        #把 idx_next 接到 idx 尾巴
        
    return idx