"""
Vector store for career documents using ChromaDB
Provides persistent storage and semantic search
"""

import chromadb
import numpy as np
from typing import List, Dict, Optional
import os
import uuid

class CareerVectorStore:
    """
    Vector store for career documents using ChromaDB.
    Provides semantic search with persistence.
    """
    
    def __init__(self, persist_directory: str = './chroma_db'):
        """
        Initialize ChromaDB with persistence.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = persist_directory
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(
                anonymized_telemetry=False
            )
        )
        
        self.collection_name = "career_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store initialized with {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict[str, str]], 
                   embeddings: np.ndarray) -> None:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: Document chunks with metadata
            embeddings: Pre-computed embeddings for each chunk
        """
        if len(chunks) == 0:
            return
        
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                'source': chunk.get('source', 'unknown'), 
                'chunk_id': chunk.get('chunk_id', str(i))
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, embedder, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant career documents.
        
        Args:
            query: Search query
            embedder: Embedder instance for generating query embeddings
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        query_embedding = embedder.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': results['distances'][0][i] if results.get('distances') else None
                }
                documents.append(doc)
        
        return documents
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata,
            'persist_directory': self.persist_directory
        }
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")
    
    def evaluate_retrieval(self, test_queries: List[str], embedder) -> Dict:
        """
        Evaluate retrieval quality with sample queries.
        
        Args:
            test_queries: List of test queries
            embedder: Embedder instance
            
        Returns:
            Evaluation results
        """
        evaluations = []
        
        for query in test_queries:
            results = self.search(query, embedder, n_results=3)
            
            query_terms = set(query.lower().split())
            relevance_scores = []
            
            for result in results:
                text = result['text'].lower()
                matches = sum(1 for term in query_terms if term in text)
                score = matches / len(query_terms) if query_terms else 0
                relevance_scores.append(score)
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            evaluations.append({
                'query': query,
                'num_results': len(results),
                'avg_relevance': avg_relevance,
                'top_result': results[0]['text'][:200] + "..." if results else "No results",
                'relevance_assessment': 'Good' if avg_relevance > 0.3 else 'Needs improvement'
            })
        
        return {
            'evaluations': evaluations,
            'summary': {
                'total_queries': len(test_queries),
                'average_relevance': sum(e['avg_relevance'] for e in evaluations) / len(evaluations) if evaluations else 0,
                'good_queries': sum(1 for e in evaluations if e['relevance_assessment'] == 'Good')
            }
        }