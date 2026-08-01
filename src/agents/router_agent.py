"""
Router Agent pattern - Routes queries to appropriate specialized handlers
Pattern: Router + Workers
Location: src/agents/router_agent.py
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import os
import re
import json
import logging

from src.agents.base_agent import BaseAgent
from src.rag.vector_store import CareerVectorStore
from src.rag.embedder import CareerEmbedder

logger = logging.getLogger(__name__)

class RouterCareerState(TypedDict):
    """State for router-based career guidance."""
    user_query: str
    messages: Annotated[List[Dict], "Messages in the conversation"]
    query_type: str
    career_path: str
    final_answer: str

class RouterCareerAgent(BaseAgent):
    """
    Router Agent pattern - Routes queries to appropriate specialized handlers.
    Pattern: Router + Worker agents.
    """
    
    def __init__(self, groq_api_key: str):
        """Initialize the router agent with Groq API key."""
        super().__init__("groq")
        self.groq_api_key = groq_api_key
        self.vector_store = CareerVectorStore()
        self.embedder = CareerEmbedder()
    
    def _router_node(self, state: RouterCareerState) -> Dict:
        """Route the query to the appropriate handler."""
        query = state["user_query"]
        
        prompt = f"""Classify the following career guidance query.

Query: {query}

Classify into exactly ONE of these categories:
1. career_comparison - Comparing multiple career paths
2. skill_assessment - Asking about skills needed
3. certification_advice - Asking about certifications
4. career_roadmap - Asking for a career progression plan
5. salary_info - Asking about salary/compensation
6. general_guidance - General career advice

Also identify if a specific IT career area is mentioned (cybersecurity, networking, development, ai/ml, devops, cloud, uiux, project_management, data_engineering, etc.)

Return ONLY a JSON-like format:
{{
    "query_type": "category_name",
    "career_area": "mentioned_career_or_null"
}}"""

        response = self.model.invoke(prompt)
        
        # Parse the response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                query_type = data.get("query_type", "general_guidance")
                career_area = data.get("career_area", None)
            else:
                query_type = "general_guidance"
                career_area = None
        except:
            query_type = "general_guidance"
            career_area = None
        
        return {
            "query_type": query_type,
            "career_path": career_area,
            "messages": [{"role": "assistant", "content": f"Routed to: {query_type}"}]
        }
    
    def _career_comparison_handler(self, state: RouterCareerState) -> Dict:
        """Handler for career comparison queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        # Search knowledge base
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Provide a detailed career comparison.

Query: {query}
Focus Area: {area if area else "General IT"}

Career Information:
{context_text}

Compare career paths based on:
1. Job responsibilities
2. Required skills and technologies
3. Certification requirements
4. Salary ranges (entry to senior)
5. Career progression
6. Job market demand
7. Work-life balance expectations
8. Entry barriers

Return a structured comparison with clear sections for each career path."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _skill_assessment_handler(self, state: RouterCareerState) -> Dict:
        """Handler for skill assessment queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Provide a detailed skill assessment and recommendations.

Query: {query}
Career Area: {area if area else "General IT"}

Career Information:
{context_text}

Provide:
1. Required technical skills (with proficiency levels)
2. Required soft skills
3. Learning resources and platforms
4. Practice projects and portfolio recommendations
5. Skill gap analysis
6. 30-60-90 day learning plan

Be specific and actionable."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _certification_handler(self, state: RouterCareerState) -> Dict:
        """Handler for certification queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Provide certification guidance.

Query: {query}
Career Area: {area if area else "General IT"}

Career Information:
{context_text}

Provide:
1. Entry-level certifications (with costs, difficulty, time to complete)
2. Intermediate certifications
3. Advanced certifications
4. Vendor-specific certifications
5. Recommended certification path (order)
6. Alternatives for different budgets
7. How to prepare for each certification

Include specific certification names and study resources."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _roadmap_handler(self, state: RouterCareerState) -> Dict:
        """Handler for career roadmap queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Create a detailed career roadmap.

Query: {query}
Career Area: {area if area else "General IT"}

Career Information:
{context_text}

Create a 5-year roadmap with:
Year 1: Foundation building
  - Skills to acquire
  - Entry-level certifications
  - First job/role types
  
Year 2: Specialization
  - Advanced skills
  - Intermediate certifications
  - Career advancement strategies

Year 3: Expertise development
  - Expert-level skills
  - Advanced certifications
  - Leadership opportunities

Year 4-5: Leadership/Senior roles
  - Management/architecture roles
  - Strategic skills

Include monthly/quarterly milestones and success metrics."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _salary_handler(self, state: RouterCareerState) -> Dict:
        """Handler for salary queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Provide salary and compensation information.

Query: {query}
Career Area: {area if area else "General IT"}

Career Information:
{context_text}

Provide:
1. Entry-level salary ranges (with location context)
2. Mid-level salary ranges
3. Senior-level salary ranges
4. Factors affecting salary (location, company size, skills)
5. Salary negotiation tips
6. Benefits and compensation packages
7. Salary growth expectations by year

Provide realistic, well-researched numbers with ranges."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _general_handler(self, state: RouterCareerState) -> Dict:
        """Handler for general career queries."""
        query = state["user_query"]
        area = state.get("career_path", "")
        
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:300] for doc in context])
        
        prompt = f"""Provide comprehensive career guidance.

Query: {query}
Career Area: {area if area else "General IT"}

Career Information:
{context_text}

Provide general guidance covering:
1. Career introduction and overview
2. Skills and qualifications
3. Career paths and progression
4. Resources for learning
5. Industry trends and outlook
6. Networking and community involvement

Make it comprehensive but structured with clear sections."""

        response = self.model.invoke(prompt)
        return {"final_answer": response.content}
    
    def _handler_router(self, state: RouterCareerState) -> str:
        """Route to appropriate handler based on query type."""
        query_type = state.get("query_type", "general_guidance")
        routing_map = {
            "career_comparison": "comparison_handler",
            "skill_assessment": "skill_handler",
            "certification_advice": "cert_handler",
            "career_roadmap": "roadmap_handler",
            "salary_info": "salary_handler",
            "general_guidance": "general_handler"
        }
        return routing_map.get(query_type, "general_handler")
    
    def create_pipeline(self):
        """Create the router-based pipeline."""
        graph = StateGraph(RouterCareerState)
        
        # Add nodes
        graph.add_node("router", self._router_node)
        graph.add_node("comparison_handler", self._career_comparison_handler)
        graph.add_node("skill_handler", self._skill_assessment_handler)
        graph.add_node("cert_handler", self._certification_handler)
        graph.add_node("roadmap_handler", self._roadmap_handler)
        graph.add_node("salary_handler", self._salary_handler)
        graph.add_node("general_handler", self._general_handler)
        
        # Set entry point
        graph.set_entry_point("router")
        
        # Conditional edges from router (Dynamic routing)
        graph.add_conditional_edges(
            "router",
            self._handler_router,
            {
                "comparison_handler": "comparison_handler",
                "skill_handler": "skill_handler",
                "cert_handler": "cert_handler",
                "roadmap_handler": "roadmap_handler",
                "salary_handler": "salary_handler",
                "general_handler": "general_handler"
            }
        )
        
        # All handlers go to END
        graph.add_edge("comparison_handler", END)
        graph.add_edge("skill_handler", END)
        graph.add_edge("cert_handler", END)
        graph.add_edge("roadmap_handler", END)
        graph.add_edge("salary_handler", END)
        graph.add_edge("general_handler", END)
        
        return graph.compile()
    
    def run(self, query: str) -> Dict:
        """Run the router agent pipeline."""
        try:
            pipeline = self.create_pipeline()
            
            initial_state = {
                "user_query": query,
                "messages": [{"role": "user", "content": query}]
            }
            
            result = pipeline.invoke(initial_state)
            
            return result
        except Exception as e:
            logger.error(f"Error in router agent: {e}")
            return {
                "final_answer": f"Error: {str(e)}",
                "messages": []
            }