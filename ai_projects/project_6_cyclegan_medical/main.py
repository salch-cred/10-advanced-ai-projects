import torch
import torch.nn as nn

class ResNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

def train():
    gen_G = ResNetGenerator()
    gen_F = ResNetGenerator()
    optimizer = torch.optim.Adam(list(gen_G.parameters()) + list(gen_F.parameters()))
    
    print("Training CycleGAN...")
    for epoch in range(2):
        domain_A = torch.randn(4, 3, 64, 64)
        fake_B = gen_G(domain_A)
        rec_A = gen_F(fake_B)
        
        loss = nn.L1Loss()(rec_A, domain_A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Cycle Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
