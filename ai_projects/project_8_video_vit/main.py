import torch
import torch.nn as nn

class VideoViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(16*16*3, 64)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x: [batch, num_frames, num_patches, patch_dim]
        b, f, p, d = x.shape
        x = x.view(b, f*p, d)
        emb = self.patch_embed(x)
        out = self.transformer(emb)
        return self.fc(out.mean(dim=1))

def train():
    model = VideoViT()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Training Video ViT...")
    for epoch in range(2):
        dummy_video = torch.randn(8, 5, 10, 16*16*3) # 8 batch, 5 frames, 10 patches
        dummy_labels = torch.randint(0, 10, (8,))
        
        optimizer.zero_grad()
        preds = model(dummy_video)
        loss = criterion(preds, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
