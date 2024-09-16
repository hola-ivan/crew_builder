# Import necessary libraries
from groq import Groq
import streamlit as st
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
import pandas as pd
import csv
import os

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
        A: The default setup allows you to create a digital strategy with autonomous agents for solopreneurs and startups. Please experiment with the input based on your needs!
        """)

# Title for the app
st.title("effiweb solutions Consulting Tool")

# API Key setup
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

# Input: Business context and challenges
business_context = st.text_area('Business Context and Challenges', help='Please describe your business context and the challenges you are facing')

# Set the number of agents dynamically based on user input
number_of_agents = st.slider('Number of Agents', min_value=2, max_value=10, value=4)

# Example data for agents
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To help solopreneurs and small businesses create effective digital strategies that align with their business goals, considering the context: {}.",
        "task": "Design a digital strategy for a business, focusing on key priorities like online presence, customer acquisition, and scaling operations. Context: {}.",
        "output": "A digital strategy plan with key initiatives in website development, social media marketing, and customer management, including budget considerations.",
        "backstory": "Tlaloc has over 10 years of experience in digital strategy, having worked with numerous startups and small businesses to enhance their online presence and growth. Context: {}."
    },
    {
        "name": "Tonantzin",
        "role": "Cloud Infrastructure Specialist",
        "goal": "To help businesses leverage cloud technology to streamline operations and reduce costs in line with the given context: {}.",
        "task": "Set up a cloud infrastructure, ensuring scalability and low operational costs based on the business needs. Context: {}.",
        "output": "A cloud architecture with clear implementation steps, cost projections, and automation features tailored to small business budgets.",
        "backstory": "Tonantzin is a certified cloud architect with a background in helping businesses transition to cloud solutions, focusing on cost efficiency and scalability. Context: {}."
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Consultant",
        "goal": "To help businesses leverage data and automation to optimize operations, focusing on customer engagement in the context: {}.",
        "task": "Implement an automated analytics system to track customer engagement and sales metrics for the business. Context: {}.",
        "output": "A dashboard providing real-time data on key performance indicators (KPIs), integrated with automated email and CRM workflows.",
        "backstory": "Arminius has a strong background in data science and automation, having implemented analytics solutions for various industries to improve customer engagement and operational efficiency. Context: {}."
    },
    {
        "name": "Thusnelda",
        "role": "Digital Marketing & Growth Consultant",
        "goal": "To help businesses scale their operations through effective digital marketing strategies, with a focus on growth within the context: {}.",
        "task": "Develop and execute a digital marketing strategy, including SEO, social media, and email campaigns for the business. Context: {}.",
        "output": "A comprehensive digital marketing plan with detailed timelines, budget allocations, and projected ROI, focusing on lead generation and brand growth.",
        "backstory": "Thusnelda is an expert in digital marketing with a proven track record of helping businesses grow their online presence and customer base through targeted marketing strategies. Context: {}."
    }
]

# Update agent details with the business context
def update_agent_goals_and_tasks(context):
    updated_agents = []
    for agent_data in example_data:
        agent_data["goal"] = agent_data["goal"].format(context)
        agent_data["task"] = agent_data["task"].format(context)
        updated_agents.append(agent_data)
    return updated_agents

# If a business context is provided, update the goals and tasks accordingly
if business_context:
    updated_data = update_agent_goals_and_tasks(business_context)
else:
    updated_data = example_data  # Default example data

namelist, rolelist, goallist, backstorylist, taskdescriptionlist, outputlist = [], [], [], [], [], []

for i in range(number_of_agents):
    agent = updated_data[i % len(updated_data)]  # Cycle through available agents if more agents are requested
    agent_name = st.text_input(f"Name of agent {i+1}", value=agent["name"])
    namelist.append(agent_name)
    rolelist.append(st.text_input(f"Role of {agent_name}", value=agent["role"]))
    goallist.append(st.text_area(f"Goal of {agent_name}", value=agent["goal"]))
    backstorylist.append(st.text_input(f"Describe the backstory of {agent_name}", value=agent["backstory"]))
    taskdescriptionlist.append(st.text_area(f"Describe the task of {agent_name}", value=agent["task"]))
    outputlist.append(st.text_area(f"Expected output of {agent_name}", value=agent["output"]))

# Create click button
if st.button('Start'):
    agentlist, tasklist_full = [], []
    for i in range(number_of_agents):
        agent = Agent(
            role=rolelist[i],
            goal=goallist[i],
            backstory=backstorylist[i],  # Ensure backstory is provided
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=True,  # Enable delegation for task optimization
            max_iter=5,
            memory=True
        )
        agentlist.append(agent)
        tasklist_full.append(Task(description=taskdescriptionlist[i], expected_output=outputlist[i], agent=agent))
    
    crew = Crew(agents=agentlist, tasks=tasklist_full, verbose=True, process=Process.sequential, full_output=True)
    # Kick off the crew's work
    results = crew.kickoff()
    # Print the results
    st.write("Crew Work Results:")
    
    for i in range(number_of_agents):
        # Display outputs dynamically for each agent
        st.subheader(f"Agent {i+1} - {namelist[i]}")
        st.write(f"**Role**: {rolelist[i]}")
        st.write(f"**Goal**: {goallist[i]}")
        st.write(f"**Task Description**: {taskdescriptionlist[i]}")
        st.write(f"**Expected Output**: {outputlist[i]}")
        st.write(f"**Actual Output**: {tasklist_full[i].output.raw}")  # Updated to show actual output
else:
    st.write('Please click the button to perform an operation')

# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")