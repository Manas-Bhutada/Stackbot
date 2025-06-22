from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.qa_pairs = []

    def load_data(self, json_path="data/qna.json"):
        with open(json_path, "r") as f:
            self.qa_pairs = json.load(f)

    def build_index(self):
        questions = [item["question"] for item in self.qa_pairs]
        embeddings = self.model.encode(questions, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        os.makedirs("embeddings", exist_ok=True)
        faiss.write_index(self.index, "embeddings/faiss_index.bin")
        with open("embeddings/qa_meta.json", "w") as f:
            json.dump(self.qa_pairs, f, indent=2)

    def load_index(self):
        self.index = faiss.read_index("embeddings/faiss_index.bin")
        with open("embeddings/qa_meta.json", "r") as f:
            self.qa_pairs = json.load(f)

    def query(self, question, top_k=3):
        embedding = self.model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(embedding, top_k)
        return [self.qa_pairs[i] for i in I[0]]
