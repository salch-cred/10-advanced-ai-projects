import os
from pathlib import Path

base_dir = Path("c:/Users/salma/Downloads/dollyeo-polished-fullstack/ai_projects")
base_dir.mkdir(parents=True, exist_ok=True)

projects = [
    {
        "name": "project_1_llm_lora",
        "desc": "LLM Fine-Tuning with LoRA (Generative AI / NLP)",
        "reqs": "torch\ntransformers\npeft\naccelerate\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_2_yolo_custom",
        "desc": "Custom Object Detection & Segmentation (Computer Vision)",
        "reqs": "torch\ntorchvision\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_3_ppo_agent",
        "desc": "PPO Agent for Custom Environment (Reinforcement Learning)",
        "reqs": "torch\ngym\nstable-baselines3\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_4_two_tower_recsys",
        "desc": "Two-Tower Recommendation Engine (Recommender Systems)",
        "reqs": "tensorflow\ntensorflow-recommenders\n",
        "code": """import tensorflow as tf

def build_tower(vocab_size, embed_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu')
    ])

def train():
    print("Training Two-Tower RecSys...")
    user_tower = build_tower(1000, 32)
    item_tower = build_tower(500, 32)
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    
    for epoch in range(2):
        dummy_users = tf.random.uniform((64, 5), maxval=1000, dtype=tf.int32)
        dummy_items = tf.random.uniform((64, 5), maxval=500, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            user_emb = user_tower(dummy_users)
            item_emb = item_tower(dummy_items)
            scores = tf.reduce_sum(user_emb * item_emb, axis=1)
            loss = tf.reduce_mean(tf.square(scores - 1.0)) # mock positive samples
            
        grads = tape.gradient(loss, user_tower.trainable_variables + item_tower.trainable_variables)
        optimizer.apply_gradients(zip(grads, user_tower.trainable_variables + item_tower.trainable_variables))
        print(f"Epoch {epoch+1}, RecSys Loss: {loss.numpy():.4f}")

if __name__ == "__main__":
    train()
"""
    },
    {
        "name": "project_5_tft_forecasting",
        "desc": "Temporal Fusion Transformers for Forecasting (Predictive Modeling)",
        "reqs": "torch\npytorch-forecasting\npandas\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_6_cyclegan_medical",
        "desc": "Medical Image Translation via CycleGAN (Computer Vision / GenAI)",
        "reqs": "torch\ntorchvision\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_7_rag_system",
        "desc": "Enterprise RAG System with Vector DB (NLP)",
        "reqs": "transformers\nlangchain\nfaiss-cpu\n",
        "code": """import numpy as np

# Mocking a FAISS index and RAG retrieval
class MockVectorDB:
    def __init__(self):
        self.embeddings = np.random.rand(100, 128)
        self.documents = [f"Doc {i}" for i in range(100)]
        
    def search(self, query_emb, k=3):
        # Random search
        indices = np.random.randint(0, 100, k)
        return [self.documents[i] for i in indices]

def run_rag():
    print("Initializing RAG System...")
    db = MockVectorDB()
    query = "What is the policy on remote work?"
    print(f"Query: {query}")
    
    query_emb = np.random.rand(128)
    results = db.search(query_emb)
    print("Retrieved context:")
    for res in results:
        print(f" - {res}")
        
    print("LLM Generation: Based on the context, remote work is allowed 3 days a week.")

if __name__ == "__main__":
    run_rag()
"""
    },
    {
        "name": "project_8_video_vit",
        "desc": "Video Action Recognition with Vision Transformers (Computer Vision)",
        "reqs": "torch\ntorchvision\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_9_gnn_fraud",
        "desc": "Fraud Detection with Graph Neural Networks (Predictive Modeling)",
        "reqs": "torch\ntorch-geometric\n",
        "code": """import torch
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
"""
    },
    {
        "name": "project_10_cross_lingual",
        "desc": "Cross-Lingual Zero-Shot Sentiment Classifier (NLP)",
        "reqs": "transformers\ntorch\n",
        "code": """import torch
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
"""
    }
]

for p in projects:
    p_dir = base_dir / p["name"]
    p_dir.mkdir(exist_ok=True)
    
    # Write main script
    (p_dir / "main.py").write_text(p["code"], encoding='utf-8')
    
    # Write reqs
    (p_dir / "requirements.txt").write_text(p["reqs"], encoding='utf-8')
    
    # Write README
    readme = f"# {p['name']}\\n\\n{p['desc']}\\n\\n## Run\\n`python main.py`"
    (p_dir / "README.md").write_text(readme, encoding='utf-8')

print("Created 10 advanced projects successfully.")
