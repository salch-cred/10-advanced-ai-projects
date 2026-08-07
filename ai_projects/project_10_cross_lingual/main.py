import torch
import torch.nn as nn

class MockXLMRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 128)
        self.fc = nn.Linear(128, 3) # Neg, Neu, Pos
        
    def forward(self, x):
        return self.fc(self.embedding(x).mean(dim=1))

def evaluate_zero_shot():
    model = MockXLMRoberta()
    print("Running Zero-Shot Cross-Lingual Evaluation...")
    
    # Mock English train loss
    print("Trained on English sentiment data...")
    
    # Evaluate on Spanish
    es_data = torch.randint(0, 5000, (10, 20))
    es_preds = model(es_data).argmax(dim=-1)
    print(f"Spanish Test Acc (mock): 85.4%")
    
    # Evaluate on Chinese
    zh_data = torch.randint(0, 5000, (10, 20))
    zh_preds = model(zh_data).argmax(dim=-1)
    print(f"Chinese Test Acc (mock): 82.1%")

if __name__ == "__main__":
    evaluate_zero_shot()
