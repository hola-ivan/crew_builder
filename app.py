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
        A: The default setup allows you to use it to create a digital strategy with autonomous agents for solopreneurs and startups. Please experiment with the input based on your needs!
        """)

# Title for the app
st.title("effiweb solutions Consulting Tool")

# API Key setup
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

# Set the number of agents to 4
number_of_agents = 4

# Example data for pre-filling the form
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To help solopreneurs and small businesses create effective digital strategies that align with their business goals.",
        "backstory": "An expert in working with small businesses, Tlaloc focuses on developing cost-effective digital roadmaps that can scale with growth.",
        "task": "Design a digital strategy for an early-stage startup, focusing on key priorities like online presence, customer acquisition, and scaling operations.",
        "output": "A digital strategy plan with key initiatives in website development, social media marketing, and customer management, including budget considerations."
    },
    {
        "name": "Tonantzin",
        "role": "Cloud Infrastructure Specialist",
        "goal": "To help startups and small businesses leverage cloud technology to streamline operations and reduce costs.",
        "backstory": "Specializing in affordable cloud solutions, Tonantzin helps clients implement cloud systems that grow with their business needs.",
        "task": "Set up a cloud infrastructure for a solopreneur’s e-commerce business, ensuring scalability and low operational costs.",
        "output": "A cloud architecture with clear implementation steps, cost projections, and automation features tailored to small business budgets."
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Consultant",
        "goal": "To help solopreneurs and startups leverage data and automation to optimize their operations and improve customer engagement.",
        "backstory": "With a background in startups, Arminius focuses on affordable analytics and automation tools that help small businesses track growth and customer behavior.",
        "task": "Implement an automated analytics system to track customer engagement and sales metrics for a small online business.",
        "output": "A dashboard providing real-time data on key performance indicators (KPIs), integrated with automated email and CRM workflows."
    },
    {
        "name": "Thusnelda",
        "role": "Digital Marketing & Growth Consultant",
        "goal": "To help small businesses and solopreneurs scale their operations through effective digital marketing strategies.",
        "backstory": "A digital marketing expert, Thusnelda has worked with early-stage startups to build brand awareness, acquire customers, and drive growth.",
        "task": "Develop and execute a digital marketing strategy, including SEO, social media, and email campaigns, for a small business looking to expand its customer base.",
        "output": "A comprehensive digital marketing plan with detailed timelines, budget allocations, and projected ROI, focusing on lead generation and brand growth."
    }
]

# Agent details collection with pre-filled data
namelist, rolelist, goallist, backstorylist, taskdescriptionlist, outputlist = [], [], [], [], [], []
for i in range(number_of_agents):
    example = example_data[i]
    with st.expander(f"Agent {i+1} Details"):
        agent_name = st.text_input(f"Name of agent {i+1}", value=example["name"])
        namelist.append(agent_name)
        rolelist.append(st.text_input(f"Role of {agent_name}", value=example["role"]))
        goallist.append(st.text_input(f"Goal of {agent_name}", value=example["goal"]))
        backstorylist.append(st.text_input(f"Backstory of {agent_name}", value=example["backstory"]))
        taskdescriptionlist.append(st.text_input(f"Task of {agent_name}", value=example["task"]))
        outputlist.append(st.text_input(f"Expected output of {agent_name}", value=example["output"]))

# Crew creation and results display
if st.button('Start'):
    agentlist, tasklist = [], []
    for i in range(number_of_agents):
        agent = Agent(role=rolelist[i], goal=goallist[i], backstory=backstorylist[i], llm=GROQ_LLM, verbose=True, max_iter=5)
        agentlist.append(agent)
        tasklist.append(Task(description=taskdescriptionlist[i], expected_output=outputlist[i], agent=agent))
    
    # Fix: verbose should be a boolean
    crew = Crew(agents=agentlist, tasks=tasklist, verbose=True, process=Process.sequential, full_output=True)
    results = crew.kickoff()

    # Summary generation using Groq API
    summary_content = "\n".join([f"{namelist[i]}({rolelist[i]}): {task.output}" for i, task in enumerate(tasklist)])
    summary_input = {"role": "user", "content": f"Summarize the following results:\n{summary_content}"}

    completion = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=[summary_input], temperature=1)
    summary_text = completion.choices[0].message.content

    # Displaying results
    st.markdown("---")
    st.subheader("Summary")
    st.markdown(summary_text)

    st.subheader("Results")
    for i in range(number_of_agents):
        st.write(f"{namelist[i]}'s ({rolelist[i]}) Output: {tasklist[i].output}")
else:
    st.write('⬆️Ready to start')


# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")
