"""
Single Agent pattern - One agent with ReAct loop handles the entire task
Pattern: ReAct (Reasoning + Acting)
Location: src/agents/single_agent.py
"""

from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import os
import logging

from src.rag.vector_store import CareerVectorStore
from src.rag.embedder import CareerEmbedder
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CareerState(TypedDict):
    """State for the career guidance agent."""
    user_query: str
    messages: Annotated[List[Dict], "Messages in the conversation"]
    search_results: str
    career_recommendation: str
    roadmaps: str
    final_answer: str

class SingleCareerAgent(BaseAgent):
    """
    Single Agent pattern - One agent handles the entire career guidance process.
    Uses ReAct (Reasoning + Acting) loop.
    """
    
    def __init__(self, groq_api_key: str):
        """Initialize the single agent with Groq API key."""
        super().__init__("groq")
        self.groq_api_key = groq_api_key
        self.vector_store = CareerVectorStore()
        self.embedder = CareerEmbedder()
        
        # System prompt for the agent
        self.system_prompt = """You are an IT Career Guidance Assistant specializing in helping people find their ideal career path in technology.

Your role is to:
1. Understand the user's background, interests, and goals
2. Research relevant IT career paths using the knowledge base
3. Provide detailed career recommendations with:
   - Required skills and technologies
   - Recommended certifications
   - Career progression path
   - Salary expectations and growth opportunities
   - Entry requirements and timeframes

Use the search tool to find career information from the knowledge base.
Always structure your answers clearly with sections.

Available IT Career Areas:
- Cybersecurity (Red Teaming, Blue Teaming, SOC Analyst, Penetration Testing, GRC)
- Networking (Network Engineer, Network Admin, Cloud Network Architect)
- Software Development (Frontend, Backend, Full Stack, Mobile)
- AI/ML (AI Engineer, Data Scientist, ML Engineer)
- DevOps/DevSecOps (DevOps Engineer, SRE, DevSecOps)
- UX/UI Design (UX Designer, UI Designer)
- Project Management (Project Manager, Product Manager, Scrum Master)
- Data Engineering (Data Engineer, Data Analyst)
- Cloud Architecture (Cloud Architect, Cloud Consultant)
"""
    
    def _create_tools(self):
        """Create tools for the agent."""
        def search_knowledge_base(query: str) -> str:
            """Search the career knowledge base."""
            try:
                results = self.vector_store.search(query, self.embedder, n_results=3)
                
                if not results:
                    return "No relevant career information found. Please try a different query."
                
                # Format results
                output = "=== Career Information ===\n\n"
                for i, result in enumerate(results, 1):
                    output += f"[{i}] {result['text']}\n\n"
                    if result.get('metadata'):
                        output += f"Source: {result['metadata'].get('source', 'Unknown')}\n\n"
                
                return output
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        return [Tool(
            name="search_knowledge_base",
            func=search_knowledge_base,
            description="Search for IT career information including roles, skills, certifications, and progression paths."
        )]
    
    def create_agent(self):
        """Create the ReAct agent."""
        tools = self._create_tools()
        return create_react_agent(
            model=self.model,
            tools=tools,
            state_modifier=self.system_prompt
        )
    
    def run(self, query: str) -> Dict:
        """Run the single agent."""
        try:
            agent = self.create_agent()
            
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # Extract the final answer
            final_answer = ""
            for message in result.get("messages", []):
                if message.get("role") == "assistant" and not message.get("tool_calls"):
                    final_answer = message.get("content", "")
                    break
            
            return {
                "final_answer": final_answer,
                "state": result
            }
        except Exception as e:
            logger.error(f"Error in single agent: {e}")
            return {
                "final_answer": f"Error: {str(e)}",
                "state": {}
            }