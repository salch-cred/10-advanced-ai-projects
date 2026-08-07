import torch
import torch.nn as nn

class SimpleTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train():
    model = SimpleTimeSeriesModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    print("Training Forecasting Model...")
    for epoch in range(2):
        # [batch, seq_len, features]
        dummy_seq = torch.randn(32, 10, 1)
        dummy_target = torch.randn(32, 1)
        
        optimizer.zero_grad()
        preds = model(dummy_seq)
        loss = criterion(preds, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Forecast Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
