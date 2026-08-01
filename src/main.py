"""
IT Career Guidance Agent - Main Orchestrator
"""

import os
from typing import Dict
from dotenv import load_dotenv
import logging

# Import agents
from src.agents.single_agent import SingleCareerAgent
from src.agents.sequential_agent import SequentialCareerAgent
from src.agents.parallel_agent import ParallelCareerAgent
from src.agents.router_agent import RouterCareerAgent

# Import RAG components
from src.rag.chunker import CareerDocumentChunker
from src.rag.embedder import CareerEmbedder
from src.rag.vector_store import CareerVectorStore

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CareerGuidanceOrchestrator:
    """Main orchestrator for the IT career guidance application."""
    
    def __init__(self):
        """Initialize the orchestrator with all components."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize RAG components
        self._init_rag()
        
        # Initialize agents
        self._init_agents()
        
        logger.info("Orchestrator initialized successfully")
    
    def _init_rag(self):
        """Initialize the RAG pipeline."""
        logger.info("Initializing RAG pipeline...")
        
        self.chunker = CareerDocumentChunker(chunk_size=500, overlap=50)
        self.embedder = CareerEmbedder()
        self.vector_store = CareerVectorStore()
        
        if self.vector_store.collection.count() == 0:
            logger.info("Populating vector store with career documents...")
            self._populate_vector_store()
        else:
            logger.info(f"Vector store already populated with {self.vector_store.collection.count()} documents")
    
    def _populate_vector_store(self):
        """Populate the vector store with career documents."""
        os.makedirs('data/knowledge_base', exist_ok=True)
        self._create_sample_documents()
        
        chunks = self.chunker.process_directory('data/knowledge_base')
        logger.info(f"Created {len(chunks)} chunks from career documents")
        
        if chunks:
            embedded_chunks = self.embedder.embed_chunks(chunks)
            self.vector_store.add_chunks(
                embedded_chunks,
                self.embedder.embed_texts([chunk['text'] for chunk in embedded_chunks])
            )
            logger.info(f"Vector store populated with {self.vector_store.collection.count()} documents")
    
    def _create_sample_documents(self):
        """Create sample career documents if they don't exist."""
        sample_docs = {
            'cybersecurity.txt': """# Cybersecurity Career Guide

## Overview
Cybersecurity professionals protect systems, networks, and data from cyber threats.

## Sub-roles
1. Red Team (Penetration Testing)
2. Blue Team (Security Operations)
3. SOC Analyst
4. GRC Specialist
5. Security Architect

## Required Skills
- Network Security
- Threat Analysis
- Incident Response
- Security Tools
- Risk Assessment

## Certifications
- CompTIA Security+
- CISSP
- CEH
- CISA
- OSCP

## Career Path
- Junior Security Analyst
- Security Engineer
- Senior Security Engineer
- Security Architect
- CISO""",
            
            'developer.txt': """# Software Development Career Guide

## Overview
Software developers design, develop, and maintain software applications.

## Sub-roles
- Frontend Developer
- Backend Developer
- Full Stack Developer
- Mobile Developer
- API Developer

## Required Skills
- Programming Languages
- Data Structures
- Version Control (Git)
- Testing/Debugging
- Design Patterns

## Certifications
- AWS Developer Associate
- Microsoft Certified: Azure Developer
- Oracle Certified Java Developer
- Google Associate Android Developer

## Career Path
- Junior Developer
- Developer
- Senior Developer
- Lead Developer
- Principal Engineer"""
        }
        
        for filename, content in sample_docs.items():
            filepath = os.path.join('data/knowledge_base', filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created sample document: {filename}")
    
    def _init_agents(self):
        """Initialize all agent patterns."""
        logger.info("Initializing agents...")
        
        self.single_agent = SingleCareerAgent(self.groq_api_key)
        self.sequential_agent = SequentialCareerAgent(self.groq_api_key)
        self.parallel_agent = ParallelCareerAgent(self.groq_api_key)
        self.router_agent = RouterCareerAgent(self.groq_api_key)
        
        logger.info("All agents initialized successfully")
    
    def run_with_pattern(self, query: str, pattern: str = "single") -> Dict:
        """Run the career guidance with a specific agent pattern."""
        pattern_map = {
            "single": self.single_agent.run,
            "sequential": self.sequential_agent.run,
            "parallel": self.parallel_agent.run,
            "router": self.router_agent.run
        }
        
        if pattern not in pattern_map:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        logger.info(f"Running {pattern} agent pattern...")
        result = pattern_map[pattern](query)
        
        return {
            "pattern": pattern,
            "query": query,
            "result": result,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }

_orchestrator = None

def get_orchestrator():
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CareerGuidanceOrchestrator()
    return _orchestrator