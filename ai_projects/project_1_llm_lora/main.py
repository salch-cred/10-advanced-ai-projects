import torch
import torch.nn as nn

# Dummy transformer and training loop
class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4), num_layers=2)
        self.fc = nn.Linear(128, 1000)

    def forward(self, x):
        return self.fc(self.transformer(self.embedding(x)))

def train():
    model = DummyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training started...")
    for epoch in range(2):
        dummy_data = torch.randint(0, 1000, (32, 10))
        dummy_targets = torch.randint(0, 1000, (32, 10))
        
        optimizer.zero_grad()
        output = model(dummy_data)
        loss = criterion(output.view(-1, 1000), dummy_targets.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
