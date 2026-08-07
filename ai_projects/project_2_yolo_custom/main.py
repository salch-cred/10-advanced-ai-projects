import torch
import torch.nn as nn

class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 4) # Bounding box

    def forward(self, x):
        return self.fc(self.cnn(x).view(-1, 32))

def train():
    model = SimpleDetector()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    print("Training detector...")
    for epoch in range(2):
        dummy_images = torch.randn(8, 3, 64, 64)
        dummy_bboxes = torch.rand(8, 4)
        
        optimizer.zero_grad()
        preds = model(dummy_images)
        loss = criterion(preds, dummy_bboxes)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
