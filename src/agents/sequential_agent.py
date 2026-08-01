"""
Sequential Agent pattern - Assembly line of specialized agents
Pattern: Sequential Pipeline
Location: src/agents/sequential_agent.py
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import os
import logging

from src.agents.base_agent import BaseAgent
from src.rag.vector_store import CareerVectorStore
from src.rag.embedder import CareerEmbedder

logger = logging.getLogger(__name__)

class SequentialCareerState(TypedDict):
    """State for sequential career guidance pipeline."""
    user_query: str
    messages: Annotated[List[Dict], "Messages in the conversation"]
    
    # Agent 1: Career Matching
    career_match: str
    
    # Agent 2: Skill Analysis
    required_skills: str
    skill_gap_analysis: str
    
    # Agent 3: Certification Recommendations
    certifications: str
    
    # Agent 4: Final Roadmap
    final_roadmap: str

class SequentialCareerAgent(BaseAgent):
    """
    Sequential Agent pattern - Assembly line of specialized agents.
    Each agent does one specific task and passes results forward.
    Pattern: Sequential Pipeline (Assembly Line)
    """
    
    def __init__(self, groq_api_key: str):
        """Initialize the sequential agent with Groq API key."""
        super().__init__("groq")
        self.groq_api_key = groq_api_key
        self.vector_store = CareerVectorStore()
        self.embedder = CareerEmbedder()
    
    def _career_matching_agent(self, state: SequentialCareerState) -> Dict:
        """Agent 1: Match user profile to best career path."""
        query = state["user_query"]
        
        # Search knowledge base first
        context = self.vector_store.search(query, self.embedder, n_results=3)
        context_text = "\n".join([doc['text'][:500] for doc in context])
        
        prompt = f"""Based on the user's query and career information, identify the most suitable IT career path.

User Query: {query}

Career Information:
{context_text}

Consider these career areas:
1. Cybersecurity (Red Team, Blue Team, SOC, Pen Testing, GRC)
2. Networking
3. Software Development
4. AI/ML
5. DevOps/DevSecOps
6. UX/UI Design
7. Project Management
8. Data Engineering
9. Cloud Architecture

Output format:
Most Suitable Career: [Career Name]
Confidence: [High/Medium/Low]
Reasons: [Brief justification]
Alternative Options: [2-3 alternatives]

Return only the analysis, no additional text."""

        response = self.model.invoke(prompt)
        
        return {
            "career_match": response.content,
            "messages": [{"role": "assistant", "content": "Career matching completed."}]
        }
    
    def _skill_analysis_agent(self, state: SequentialCareerState) -> Dict:
        """Agent 2: Analyze required skills and skill gaps."""
        career_match = state.get("career_match", "")
        
        prompt = f"""Based on the identified career path, analyze the skills required.

Identified Career: {career_match}

Provide:
1. Required Technical Skills (list with proficiency levels - Beginner/Intermediate/Advanced)
2. Required Soft Skills
3. Current skill gaps (assume entry level for now)
4. Learning path recommendations
5. Estimated time to acquire skills (3-6 months, 6-12 months, 1-2 years)

Format with clear sections. Be specific and actionable.

Return only the skills analysis."""

        response = self.model.invoke(prompt)
        
        return {
            "required_skills": response.content,
            "messages": [{"role": "assistant", "content": "Skills analysis completed."}]
        }
    
    def _certification_agent(self, state: SequentialCareerState) -> Dict:
        """Agent 3: Recommend certifications for the career path."""
        career_match = state.get("career_match", "")
        skills = state.get("required_skills", "")
        
        prompt = f"""Recommend relevant certifications for the career path.

Career: {career_match}
Skills Context: {skills}

Provide:
1. Entry-level certifications (with difficulty, cost estimates, and time to complete)
2. Intermediate certifications
3. Advanced certifications
4. Vendor-specific certifications (AWS, Microsoft, Cisco, CompTIA, etc.)
5. Recommended order to pursue certifications
6. Alternatives for different budget levels

Return only the certification recommendations with clear categories."""

        response = self.model.invoke(prompt)
        
        return {
            "certifications": response.content,
            "messages": [{"role": "assistant", "content": "Certification recommendations completed."}]
        }
    
    def _roadmap_agent(self, state: SequentialCareerState) -> Dict:
        """Agent 4: Create final career roadmap."""
        career = state.get("career_match", "")
        skills = state.get("required_skills", "")
        certs = state.get("certifications", "")
        
        prompt = f"""Create a comprehensive career roadmap.

Career: {career}
Skills Required: {skills}
Certifications: {certs}

Create a 3-5 year roadmap with:
Year 1: Entry-level goals
  - Skills to learn
  - Certifications to get
  - First job targets
  
Year 2: Intermediate goals
  - Advanced skills
  - Professional growth
  - Career advancement
  
Year 3: Advanced goals
  - Expert-level skills
  - Leadership opportunities
  
Year 4+: Expert/Leadership goals
  - Senior positions
  - Industry impact

Include:
- Monthly/quarterly milestones
- Skill acquisition targets
- Certification timeline
- Experience requirements
- Portfolio/project recommendations
- Networking and community involvement

Make it realistic, specific, and actionable.

Return only the roadmap with clear timelines."""

        response = self.model.invoke(prompt)
        
        return {
            "final_roadmap": response.content,
            "messages": [{"role": "assistant", "content": "Roadmap completed."}]
        }
    
    def create_pipeline(self):
        """Create the sequential pipeline."""
        graph = StateGraph(SequentialCareerState)
        
        # Add nodes
        graph.add_node("career_matcher", self._career_matching_agent)
        graph.add_node("skill_analyst", self._skill_analysis_agent)
        graph.add_node("certification_specialist", self._certification_agent)
        graph.add_node("roadmap_builder", self._roadmap_agent)
        
        # Set entry point
        graph.set_entry_point("career_matcher")
        
        # Sequential edges (Assembly Line)
        graph.add_edge("career_matcher", "skill_analyst")
        graph.add_edge("skill_analyst", "certification_specialist")
        graph.add_edge("certification_specialist", "roadmap_builder")
        graph.add_edge("roadmap_builder", END)
        
        return graph.compile()
    
    def run(self, query: str) -> Dict:
        """Run the sequential agent pipeline."""
        try:
            pipeline = self.create_pipeline()
            
            initial_state = {
                "user_query": query,
                "messages": [{"role": "user", "content": query}]
            }
            
            result = pipeline.invoke(initial_state)
            
            return result
        except Exception as e:
            logger.error(f"Error in sequential agent: {e}")
            return {
                "final_roadmap": f"Error: {str(e)}",
                "messages": []
            }