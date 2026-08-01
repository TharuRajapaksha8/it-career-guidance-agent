"""
Document chunking strategy for RAG pipeline
Implements section-based chunking with overlap
"""

import re
from typing import List, Dict
import os

class CareerDocumentChunker:
    """
    Chunking strategy for career guidance documents.
    Uses section-based splitting with overlap for context preservation.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_and_chunk(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load a career document and split into chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of chunks with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._chunk_content(content, file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _chunk_content(self, content: str, source: str) -> List[Dict[str, str]]:
        """
        Split content into overlapping chunks.
        
        Args:
            content: Document content
            source: Source file name
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        sections = re.split(r'\n#{1,3}\s+', content)
        sections = [s.strip() for s in sections if s.strip()]
        
        for section in sections:
            if len(section) > self.chunk_size:
                words = section.split()
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_text = ' '.join(words[i:i + self.chunk_size])
                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'source': os.path.basename(source),
                            'chunk_id': f"{os.path.basename(source)}_{len(chunks)}"
                        })
            else:
                if section.strip():
                    chunks.append({
                        'text': section,
                        'source': os.path.basename(source),
                        'chunk_id': f"{os.path.basename(source)}_{len(chunks)}"
                    })
        
        return chunks
    
    def process_directory(self, dir_path: str) -> List[Dict[str, str]]:
        """
        Process all career documents in a directory.
        
        Args:
            dir_path: Directory containing career documents
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            return all_chunks
            
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(dir_path, file_name)
                chunks = self.load_and_chunk(file_path)
                all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {dir_path}")
        return all_chunks