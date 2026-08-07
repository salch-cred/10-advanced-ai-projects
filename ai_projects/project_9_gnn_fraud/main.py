import torch
import torch.nn as nn

# Mocking a simple GNN without pyg for dependency simplicity in script
class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 2)
        
    def forward(self, x, adj):
        # x: [num_nodes, 32], adj: [num_nodes, num_nodes]
        h = torch.relu(self.linear1(adj @ x))
        return self.linear2(adj @ h)

def train():
    model = SimpleGNN()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Training GNN for Fraud Detection...")
    for epoch in range(2):
        num_nodes = 100
        dummy_features = torch.randn(num_nodes, 32)
        dummy_adj = (torch.rand(num_nodes, num_nodes) > 0.9).float()
        dummy_labels = torch.randint(0, 2, (num_nodes,))
        
        optimizer.zero_grad()
        preds = model(dummy_features, dummy_adj)
        loss = criterion(preds, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, GNN Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
