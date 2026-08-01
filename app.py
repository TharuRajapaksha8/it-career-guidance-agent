"""
IT Career Guidance Agent - Streamlit Application
Location: app.py
"""

import streamlit as st
import os
import sys
import logging

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import orchestrator
from src.main import get_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="IT Career Guidance AI Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stChatMessage {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
    }
    .pattern-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        background: #e8f5e9;
        color: #2e7d32;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .sidebar-section {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """👋 Welcome to the IT Career Guidance AI Agent!

I can help you with:
- 📋 **Role Information** - Detailed info about IT careers
- 🗺️ **Career Paths** - Progression and advancement
- 📜 **Certifications** - Recommended certifications
- 💡 **Skill Assessment** - Skills needed and gaps
- 📊 **Career Comparison** - Compare different paths
- 💰 **Salary Info** - Compensation and growth

**Try asking me:**
- "What skills do I need for cybersecurity?"
- "Tell me about DevOps career path"
- "Compare AI/ML vs Data Engineering"
- "What certifications for cloud architect?"

Select a pattern from the sidebar to see different agent behaviors!"""}
    ]

if 'initialized' not in st.session_state:
    with st.spinner("🔄 Initializing AI Agents and Knowledge Base..."):
        try:
            st.session_state.orchestrator = get_orchestrator()
            st.session_state.initialized = True
            st.success("✅ Agents initialized successfully!")
            logger.info("Application initialized successfully")
        except Exception as e:
            st.error(f"❌ Error initializing: {str(e)}")
            logger.error(f"Initialization error: {e}")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("💼 Career Guide")
    
    st.markdown("---")
    
    # Pattern selection
    st.markdown("### 🤖 Agent Pattern")
    pattern = st.selectbox(
        "Select agent pattern:",
        ["single", "sequential", "parallel", "router"],
        help="Different agent patterns for different use cases",
        index=0
    )
    st.session_state.pattern = pattern
    
    # Pattern info
    pattern_info = {
        "single": "🧠 One agent with ReAct loop. Best for simple, flexible tasks.",
        "sequential": "📋 Assembly line of agents. Best for ordered, structured workflows.",
        "parallel": "⚡ Multiple agents researching simultaneously. Best for independent tasks.",
        "router": "🎯 Router classifies query, sends to specialist. Best for diverse query types."
    }
    st.info(pattern_info.get(pattern, ""))
    
    st.markdown("---")
    
    # Quick role selectors
    st.markdown("### 🎯 Quick Role Guide")
    roles = [
        "🛡️ Cybersecurity",
        "🌐 Networking",
        "💻 Software Developer",
        "🔧 DevOps/DevSecOps",
        "☁️ Cloud Architecture",
        "🤖 AI/ML Engineer",
        "🎨 UI/UX Designer",
        "📊 Project Manager",
        "📈 Data Engineer"
    ]
    
    for role in roles:
        if st.button(role, key=f"quick_{role}"):
            query = f"Tell me about {role}"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.markdown("---")
    
    # System info
    with st.expander("📊 About This System"):
        st.markdown("""
        **Agent Patterns Used:**
        - 🎯 Router Pattern
        - 🏗️ Orchestrator-Worker
        - ⚡ Parallel Processing
        - 📊 Sequential Pipeline
        - 🔄 ReAct Loop
        
        **Models:**
        - 🚀 Groq (Llama 3 70B) - Fast, cheap for classification
        - 🧠 OpenRouter (Claude 3.5) - High quality for synthesis
        
        **RAG Pipeline:**
        - 📚 ChromaDB
        - 🔍 Semantic Search
        - 📄 20+ Career Documents
        
        **GitHub:** [Repository](https://github.com/TharuRajapaksha8/it-career-guidance-agent)
        """)
    
    # Deployment info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.7rem;">
        🚀 Deployed on Streamlit Cloud<br>
        📅 August 2026
    </div>
    """, unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 IT Career Guidance Agent</h1>
    <p>AI-powered career guidance for IT professionals | Powered by Groq + OpenRouter</p>
</div>
""", unsafe_allow_html=True)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about IT roles, career paths, certifications..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your query..."):
            try:
                # Get pattern from session state
                pattern = st.session_state.get("pattern", "single")
                
                # Show which pattern is being used
                st.markdown(f"*Using pattern: **{pattern}***")
                
                # Run the orchestrator
                result = st.session_state.orchestrator.run_with_pattern(
                    prompt, 
                    pattern=pattern
                )
                
                # Extract the final answer based on pattern
                if pattern == "single":
                    answer = result['result'].get('final_answer', 'No answer generated')
                elif pattern == "sequential":
                    answer = result['result'].get('final_roadmap', 'No roadmap generated')
                elif pattern == "parallel":
                    answer = result['result'].get('final_comparison', 'No comparison generated')
                elif pattern == "router":
                    answer = result['result'].get('final_answer', 'No answer generated')
                else:
                    answer = str(result['result'])
                
                # Display response
                st.markdown(answer)
                
                # Show pattern badge
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; font-size: 0.9rem; margin-top: 10px;">
                    🤖 <strong>Pattern:</strong> {pattern} | 
                    ⏱️ <strong>Time:</strong> {result.get('timestamp', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
                
                # Add to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"❌ Error processing your request: {str(e)}")
                logger.error(f"Processing error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ I encountered an error: {str(e)}"}
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    🚀 Built with Streamlit, LangGraph, LangChain, Groq & OpenRouter | IT Career Guidance Agent
</div>
""", unsafe_allow_html=True)