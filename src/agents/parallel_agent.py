"""
Parallel Agent pattern - Multiple agents research different career paths simultaneously
Pattern: Parallel Fan-out/Fan-in
Location: src/agents/parallel_agent.py
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import os
import time
import logging

from src.agents.base_agent import BaseAgent
from src.rag.vector_store import CareerVectorStore
from src.rag.embedder import CareerEmbedder

logger = logging.getLogger(__name__)

class ParallelCareerState(TypedDict):
    """State for parallel career research."""
    user_query: str
    messages: Annotated[List[Dict], "Messages in the conversation"]
    
    # Parallel research results
    cybersecurity_research: str
    development_research: str
    devops_research: str
    networking_research: str
    cloud_research: str
    ai_ml_research: str
    
    # Aggregated result
    final_comparison: str

class ParallelCareerAgent(BaseAgent):
    """
    Parallel Agent pattern - Multiple agents research different career paths simultaneously.
    Then an aggregator synthesizes the results.
    Pattern: Parallel Fan-out/Fan-in
    """
    
    def __init__(self, groq_api_key: str):
        """Initialize the parallel agent with Groq API key."""
        super().__init__("groq")
        self.groq_api_key = groq_api_key
        self.vector_store = CareerVectorStore()
        self.embedder = CareerEmbedder()
    
    def _create_research_agent(self, career_area: str, state_key: str):
        """Factory function to create research agents."""
        def research_agent(state: ParallelCareerState) -> Dict:
            query = state["user_query"]
            
            # Search knowledge base
            context = self.vector_store.search(f"{career_area} {query}", self.embedder, n_results=2)
            context_text = "\n".join([doc['text'][:300] for doc in context])
            
            prompt = f"""Research the {career_area} career path based on the user's query.

User Query: {query}

Career Information:
{context_text}

Provide detailed information about:
1. Job roles and specializations within {career_area}
2. Required technical skills (specific technologies)
3. Required certifications
4. Typical salary ranges
5. Career progression path
6. Job market outlook
7. Entry requirements and difficulty
8. Recommended learning resources

Format with clear sections. Be specific with names of technologies, certifications, and roles.

Return only the research findings for {career_area}."""

            response = self.model.invoke(prompt)
            
            return {state_key: response.content}
        
        return research_agent
    
    def _create_aggregator(self):
        """Aggregator agent that synthesizes all parallel results."""
        def aggregator_agent(state: ParallelCareerState) -> Dict:
            cybersec = state.get("cybersecurity_research", "")
            dev = state.get("development_research", "")
            devops = state.get("devops_research", "")
            networking = state.get("networking_research", "")
            cloud = state.get("cloud_research", "")
            ai_ml = state.get("ai_ml_research", "")
            
            prompt = f"""You are a career advisor. Compare and synthesize the following career research results.

User Query: {state["user_query"]}

Cybersecurity Research:
{cybersec}

Development Research:
{dev}

DevOps Research:
{devops}

Networking Research:
{networking}

Cloud Research:
{cloud}

AI/ML Research:
{ai_ml}

Based on this research, create a comprehensive career comparison and recommendation:

1. Best Fit: Which career path seems most suitable and why?
2. Comparison Matrix: Compare key aspects (skills, salary, difficulty, growth)
3. Recommendations: Provide detailed guidance for the top 2-3 career paths
4. Next Steps: Actionable steps for the user to pursue
5. Resources: Recommended learning platforms and certifications

Make your analysis personalized and specific. Return a complete career guidance response."""

            response = self.model.invoke(prompt)
            
            return {
                "final_comparison": response.content,
                "messages": [{"role": "assistant", "content": "Career comparison completed."}]
            }
        
        return aggregator_agent
    
    def create_pipeline(self):
        """Create the parallel pipeline."""
        graph = StateGraph(ParallelCareerState)
        
        # Router node for fan-out
        def router(state: ParallelCareerState) -> Dict:
            return {"messages": [{"role": "assistant", "content": "Starting parallel research..."}]}
        
        # Add all nodes
        graph.add_node("router", router)
        
        graph.add_node("cybersec", self._create_research_agent("Cybersecurity", "cybersecurity_research"))
        graph.add_node("development", self._create_research_agent("Development", "development_research"))
        graph.add_node("devops", self._create_research_agent("DevOps", "devops_research"))
        graph.add_node("networking", self._create_research_agent("Networking", "networking_research"))
        graph.add_node("cloud", self._create_research_agent("Cloud", "cloud_research"))
        graph.add_node("ai_ml", self._create_research_agent("AI/ML", "ai_ml_research"))
        
        graph.add_node("aggregator", self._create_aggregator())
        
        # Set entry point
        graph.set_entry_point("router")
        
        # Fan-out: router to all research agents (Parallel execution)
        graph.add_edge("router", "cybersec")
        graph.add_edge("router", "development")
        graph.add_edge("router", "devops")
        graph.add_edge("router", "networking")
        graph.add_edge("router", "cloud")
        graph.add_edge("router", "ai_ml")
        
        # Fan-in: all research agents to aggregator
        graph.add_edge("cybersec", "aggregator")
        graph.add_edge("development", "aggregator")
        graph.add_edge("devops", "aggregator")
        graph.add_edge("networking", "aggregator")
        graph.add_edge("cloud", "aggregator")
        graph.add_edge("ai_ml", "aggregator")
        
        graph.add_edge("aggregator", END)
        
        return graph.compile()
    
    def run(self, query: str) -> Dict:
        """Run the parallel agent pipeline."""
        try:
            pipeline = self.create_pipeline()
            
            initial_state = {
                "user_query": query,
                "messages": [{"role": "user", "content": query}]
            }
            
            # Time the execution
            start_time = time.time()
            result = pipeline.invoke(initial_state)
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            
            return result
        except Exception as e:
            logger.error(f"Error in parallel agent: {e}")
            return {
                "final_comparison": f"Error: {str(e)}",
                "messages": [],
                "execution_time": 0
            }