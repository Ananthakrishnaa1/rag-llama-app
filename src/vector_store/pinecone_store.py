from pinecone import Pinecone
from typing import List

class PineconeStore:
    def __init__(self, api_key: str, index_name: str = "quickstart"):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Get or create index
        try:
            self.index = self.pc.Index(index_name)
        except Exception:
            # Create index if it doesn't exist
            self.pc.create_index(
                name=index_name,
                dimension=3072,  # adjust based on your LLama embeddings dimension
                metric="cosine"
            )
            self.index = self.pc.Index(index_name)

    def insert(self, embeddings: List[List[float]], texts: List[str]):
        vectors = [(str(i), emb, {"text": text}) 
                  for i, (emb, text) in enumerate(zip(embeddings, texts))]
        self.index.upsert(vectors=vectors)

    def get_collection_stats(self):
        stats = self.index.describe_index_stats()
        # Get some sample vectors
        query_response = self.index.query(
            vector=[0.0] * 3072,  # adjust dimension as needed
            top_k=5,
            include_metadata=True
        )
        samples = [{"id": match.id, "text": match.metadata["text"]} 
                  for match in query_response.matches]
        
        return {
            "name": self.index_name,
            "count": stats.total_vector_count,
            "samples": samples
        }