import numpy as np

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
