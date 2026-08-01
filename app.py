import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import get_orchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """👋 Welcome to the IT Career Guidance AI Agent!

I can help you with:
- 📋 Role Information
- 🗺️ Career Paths
- 📜 Certifications
- 💡 Skill Assessment
- 📊 Career Comparison

**Ask me anything about IT careers!**"""}
    ]

if 'initialized' not in st.session_state:
    with st.spinner("🔄 Initializing AI Agents..."):
        try:
            st.session_state.orchestrator = get_orchestrator()
            st.session_state.initialized = True
            st.success("✅ Agents initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing: {str(e)}")
            logger.error(f"Initialization error: {e}")

# Sidebar
with st.sidebar:
    st.title("💼 Career Guide")
    st.markdown("---")
    
    st.markdown("### 🤖 Agent Pattern")
    pattern = st.selectbox(
        "Select pattern:",
        ["single", "sequential", "parallel", "router"],
        index=0
    )
    st.session_state.pattern = pattern
    
    st.markdown("---")
    
 
    st.markdown("### 🎯 Quick Role Guide")
    roles = [
        "🛡️ Cybersecurity",
        "🌐 Networking",
        "💻 Developer",
        "🔧 DevOps",
        "☁️ Cloud",
        "🤖 AI/ML"
    ]
    
    for role in roles:
        if st.button(role, key=f"quick_{role}"):
            query = f"Tell me about {role}"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 IT Career Guidance Agent</h1>
    <p>AI-powered career guidance for IT professionals</p>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about IT careers..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your query..."):
            try:
                pattern = st.session_state.get("pattern", "single")
                result = st.session_state.orchestrator.run_with_pattern(prompt, pattern=pattern)
                
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
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Processing error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with Streamlit, LangGraph, Groq & OpenRouter
</div>
""", unsafe_allow_html=True)