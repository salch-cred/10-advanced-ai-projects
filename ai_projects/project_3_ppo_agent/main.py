import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

def train_ppo():
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters())
    
    print("Training PPO Agent (Mocked)...")
    for epoch in range(2):
        dummy_state = torch.randn(10, 4)
        dummy_action_probs = model(dummy_state)
        # Mock loss
        loss = -torch.log(dummy_action_probs[:, 0]).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, PPO Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_ppo()
