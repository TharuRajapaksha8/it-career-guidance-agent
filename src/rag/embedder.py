"""
Embedding model for career documents
Uses sentence-transformers for semantic search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import os

class CareerEmbedder:
    """
    Embedding model for career documents.
    Uses 'all-MiniLM-L6-v2' which is fast and efficient.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True)
    
    def embed_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict]:
        """
        Embed a list of chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks
            
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        print(f"Embedded {len(chunks)} chunks")
        return chunks
    
    def get_model_info(self) -> Dict:
        """
        Return model information for documentation.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'dimension': 384,
            'description': 'Sentence-BERT model for semantic search',
            'performance': 'Fast and efficient for production use'
        }