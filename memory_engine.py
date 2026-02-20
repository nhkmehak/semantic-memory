
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class SemanticMemoryEngine:
    def __init__(self, similarity_threshold=0.80):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.memory_store = []
        self.similarity_threshold = similarity_threshold

    # Convert sentence to vector
    def get_embedding(self, text):
        return self.model.encode(text)

    # Cosine similarity (manual implementation)
    def cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2)

    # Store or match memory
    def store_or_match(self, new_text):
        new_embedding = self.get_embedding(new_text)

        best_match = None
        highest_score = 0

        for memory in self.memory_store:
            score = self.cosine_similarity(new_embedding, memory["embedding"])

            if score > highest_score:
                highest_score = score
                best_match = memory

        # If similarity high enough → treat as same memory
        if highest_score >= self.similarity_threshold:
            best_match["frequency"] += 1
            best_match["last_accessed"] = datetime.now()
            return {
                "status": "matched",
                "matched_text": best_match["text"],
                "similarity": round(float(highest_score), 3),
                "frequency": best_match["frequency"]
            }

        # Otherwise store as new memory
        self.memory_store.append({
            "text": new_text,
            "embedding": new_embedding,
            "frequency": 1,
            "created_at": datetime.now(),
            "last_accessed": datetime.now()
        })

        return {
            "status": "stored",
            "similarity": round(float(highest_score), 3)
        }

    # Retrieve top N similar memories
    def retrieve(self, query, top_n=3):
        query_embedding = self.get_embedding(query)

        scored_memories = []

        for memory in self.memory_store:
            score = self.cosine_similarity(query_embedding, memory["embedding"])
            scored_memories.append((score, memory["text"]))

        scored_memories.sort(reverse=True)

        return scored_memories[:top_n]