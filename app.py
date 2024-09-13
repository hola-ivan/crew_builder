# Import necessary libraries
from groq import Groq
import streamlit as st
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
import pandas as pd
import csv
import os
from fpdf import FPDF  # PDF generation

# Set page configuration
st.set_page_config(page_title="effiweb solutions Consulting Tool", layout="wide")

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        body, input, button, textarea { font-family: 'Inter', sans-serif; }
        .main { background-color: #f5f7fa; }
        .stButton>button { background-color: #4f46e5; color: #fff; border-radius: 5px; padding: 10px 20px; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { color: #1e3a8a; background-color: #e0f2fe; border-radius: 5px; padding: 5px; }
        .stTextInput>div>div>label, .stTextArea>div>div>label { color: #3b82f6; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for contact, about, and FAQ sections
with st.sidebar:
    with st.expander("Contact", expanded=True):
        st.write("For inquiries: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")
    with st.expander("About", expanded=False):
        st.write("""
        The effiweb solutions **Consulting Tool** helps build autonomous agent crews to simulate a consulting team.
        **Disclaimer**: This tool is for educational purposes and not for professional consulting.
        """)
    with st.expander("FAQ", expanded=False):
        st.write("""
        **Q: How to use this tool?**  
        A: The default setup allows you to use it to create a digital strategy with autonomous agents for solopreneurs and startups. Please experiment with the input based on your needs!
        """)

# Title for the app
st.title("effiweb solutions Consulting Tool")

# API Key setup
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

# Input: Business context and challenges
business_context = st.text_area('Business Context and Challenges', help='Please describe your business context and the challenges you are facing')

# Set the number of agents to 4
number_of_agents = 4

# Example data for agents
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal_template": "To help solopreneurs and small businesses create effective digital strategies that align with their business goals, considering the context: {}.",
        "task_template": "Design a digital strategy for a business, focusing on key priorities like online presence, customer acquisition, and scaling operations. Context: {}.",
        "output": "A digital strategy plan with key initiatives in website development, social media marketing, and customer management, including budget considerations."
    },
    {
        "name": "Tonantzin",
        "role": "Cloud Infrastructure Specialist",
        "goal_template": "To help businesses leverage cloud technology to streamline operations and reduce costs in line with the given context: {}.",
        "task_template": "Set up a cloud infrastructure, ensuring scalability and low operational costs based on the business needs. Context: {}.",
        "output": "A cloud architecture with clear implementation steps, cost projections, and automation features tailored to small business budgets."
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Consultant",
        "goal_template": "To help businesses leverage data and automation to optimize operations, focusing on customer engagement in the context: {}.",
        "task_template": "Implement an automated analytics system to track customer engagement and sales metrics for the business. Context: {}.",
        "output": "A dashboard providing real-time data on key performance indicators (KPIs), integrated with automated email and CRM workflows."
    },
    {
        "name": "Thusnelda",
        "role": "Digital Marketing & Growth Consultant",
        "goal_template": "To help businesses scale their operations through effective digital marketing strategies, with a focus on growth within the context: {}.",
        "task_template": "Develop and execute a digital marketing strategy, including SEO, social media, and email campaigns for the business. Context: {}.",
        "output": "A comprehensive digital marketing plan with detailed timelines, budget allocations, and projected ROI, focusing on lead generation and brand growth."
    }
]

# Update agent details with the business context
def update_agent_goals_and_tasks(context):
    updated_agents = []
    for agent_data in example_data:
        agent_data["goal"] = agent_data["goal_template"].format(context)
        agent_data["task"] = agent_data["task_template"].format(context)
        updated_agents.append(agent_data)
    return updated_agents

# If a business context is provided, update the goals and tasks accordingly
if business_context:
    updated_data = update_agent_goals_and_tasks(business_context)
else:
    updated_data = example_data  # Default example data

# Set the number of agents to 4
number_of_agents = 4

# Display agent details with updated goals and tasks
namelist, rolelist, goallist, tasklist, outputlist = [], [], [], [], []
for i in range(number_of_agents):
    agent = updated_data[i]
    with st.expander(f"Agent {i+1} Details"):
        agent_name = st.text_input(f"Name of agent {i+1}", value=agent["name"])
        namelist.append(agent_name)
        rolelist.append(st.text_input(f"Role of {agent_name}", value=agent["role"]))
        goallist.append(st.text_area(f"Goal of {agent_name}", value=agent["goal"]))
        tasklist.append(st.text_area(f"Task of {agent_name}", value=agent["task"]))
        outputlist.append(st.text_area(f"Expected output of {agent_name}", value=agent["output"]))

# Crew creation and results display
if st.button('Start'):
    agentlist, tasklist_full = [], []
    for i in range(number_of_agents):
        agent = Agent(role=rolelist[i], goal=goallist[i], llm=GROQ_LLM, verbose=True)
        agentlist.append(agent)
        tasklist_full.append(Task(description=tasklist[i], expected_output=outputlist[i], agent=agent))
    
    crew = Crew(agents=agentlist, tasks=tasklist_full, verbose=True, process=Process.sequential, full_output=True)
    results = crew.kickoff()

    # Displaying results
    st.markdown("---")
    st.subheader("Results")
    for i in range(number_of_agents):
        st.write(f"{namelist[i]}'s ({rolelist[i]}) Output: {tasklist_full[i].output}")
else:
    st.write('⬆️ Ready to start')

# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")
