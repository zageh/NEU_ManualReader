import tiktoken
import time
import torch.nn.functional as F

def train_model(model, dataloader, optimizer, device, num_epochs=3):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch + 1}, avg_loss = {avg_loss:.4f}")
        
    return model