# IT Career Guidance Agentic AI Application

A smart AI system that helps people find the right IT career path. Just ask a question and get personalized career advice.

---

### Project Description

## What Does This App Do?

This app helps IT students and professionals figure out their career path. You can ask questions like:

- "What skills do I need for cybersecurity?"
- "What certifications should I get for cloud architecture?"
- "What's the career path for a DevOps engineer?"
- "Compare AI/ML vs Data Engineering"

### Purpose

Many people don't know what career path to choose in IT. There are so many options - cybersecurity, networking, development, cloud, AI, and more. This app uses AI agents to research your question and give you clear, helpful answers.

## Target Users

- IT students exploring career options
- Professionals wanting to switch careers
- Anyone curious about IT roles

## Architecture Diagram
<img width="608" height="873" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/a7f33bb2-e4bb-4610-83d7-0b331febe1da" />




## Agent Communication Diagram
<img width="602" height="703" alt="Agent_Communication_Diagram" src="https://github.com/user-attachments/assets/8af7b124-6000-4a77-97e9-21f1c0647cf2" />


## Agentic Design Patterns I Used
| **Single Agent (ReAct)** | One agent thinks, searches, and decides what to do next | `src/agents/single_agent.py` |
| **Sequential Agent** | Agents work like an assembly line. (Agent 1 → Agent 2 → Agent 3 → Agent 4) | `src/agents/sequential_agent.py` |
| **Parallel Agent** | Multiple agents research different topics at the same time | `src/agents/parallel_agent.py` |
| **Router Agent** | Figures out what user asking and sends it to the right specialist | `src/agents/router_agent.py` |
| **Orchestrator-Worker** | One main controller manages all other agents | `src/main.py` |


## Model Selection Strategy
| Task                            | Model (Provider)    |             Why I Chose It                                    |

| **Understanding user question** | Groq                | Very fast, free, good enough for simple tasks                 |
| **Finding career information**  | Groq                | Quick response, cheap, gives accurate infprmation             |
| **Analyzing career paths**      | Groq                | Fast enough for real-time use                                 |
| **Writing the final answer**    | OpenRouter          | Better at understanding context, gives high quality answers   |
| **Comparing careers**           | OpenRouter          | Best at combining information from many sources               |

## RAG Pipeline (Knowledge Base)
The Knowledge Base
I created career documents about IT roles:

- Cybersecurity (Red Team, Blue Team, SOC, GRC)
- Networking (Network Engineer, Admin, Architect)
- Software Development (Frontend, Backend, Full Stack)
- DevOps and DevSecOps
- Cloud Architecture
- AI/ML Engineering
- UI/UX Design
- Project Management
- Data Engineering

Each document covers:
- What the role does
- Required skills and the path
- Recommended certifications
- Career progression
- Salary ranges
  

