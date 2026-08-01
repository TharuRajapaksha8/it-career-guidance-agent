"""
Base agent class providing common functionality
Supports multiple models from Groq and OpenRouter
"""

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base agent class providing common functionality.
    Supports model selection with cost/latency trade-offs.
    """
    
    def __init__(self, model_name: str = "groq"):
        """
        Initialize the agent with selected model.
        
        Args:
            model_name: "groq" for fast/cheap, "openrouter" for high-quality reasoning
        """
        self.model_name = model_name
        self.model = self._get_model(model_name)
        logger.info(f"Initialized BaseAgent with model: {model_name}")
    
    def _get_model(self, model_name: str):
        """
        Get appropriate model based on name.
        
        Model Selection Strategy:
        - Groq: Fast, cheap, good for classification and simple tasks
        - OpenRouter: High quality, better reasoning for synthesis
        
        Args:
            model_name: "groq" or "openrouter"
            
        Returns:
            LangChain chat model instance
        """
        try:
            if model_name == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                return ChatGroq(
                    temperature=0.1,
                    model="llama3-70b-8192",
                    groq_api_key=api_key
                )
            elif model_name == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
                return ChatOpenAI(
                    temperature=0.1,
                    model="anthropic/claude-3.5-sonnet",
                    openai_api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                # Default to Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                return ChatGroq(
                    temperature=0.1,
                    model="llama3-70b-8192",
                    groq_api_key=api_key
                )
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise