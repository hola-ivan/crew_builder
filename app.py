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
business_context = st.text_area('Describe your business context and the challenges you are facing')

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
    # Other agent data...
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
if st.button('Create Crew'):
    agentlist, tasklist = [], []
    for i in range(number_of_agents):
        agent = Agent(role=rolelist[i], goal=goallist[i], backstory=backstorylist[i], llm=GROQ_LLM, verbose=True, max_iter=5)
        agentlist.append(agent)
        tasklist.append(Task(description=taskdescriptionlist[i], expected_output=outputlist[i], agent=agent))
    
    crew = Crew(agents=agentlist, tasks=tasklist, verbose=True, process=Process.sequential, full_output=True)
    results = crew.kickoff()

    # Summarize results using Groq API
    summary_content = "\n".join([f"{namelist[i]} ({rolelist[i]}): {task.output}" for i, task in enumerate(tasklist)])
    summary_input = {"role": "user", "content": f"Summarize the following results, considering the client's business context: {business_context}\n{summary_content}"}

    completion = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=[summary_input], temperature=1)
    summary_text = completion.choices[0].message.content

    # Display results and summary
    st.markdown("---")
    st.subheader("Summary")
    st.markdown(summary_text)

    st.subheader("Results")
    for i in range(number_of_agents):
        st.write(f"{namelist[i]}'s ({rolelist[i]}) Output: {tasklist[i].output}")

    # PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add business context
    pdf.cell(200, 10, txt="Business Context and Challenges", ln=True, align='L')
    pdf.multi_cell(0, 10, business_context)

    # Add summary
    pdf.cell(200, 10, txt="Summary", ln=True, align='L')
    pdf.multi_cell(0, 10, summary_text)

    # Add individual agent outputs
    pdf.cell(200, 10, txt="Results", ln=True, align='L')
    for i in range(number_of_agents):
        pdf.cell(200, 10, txt=f"{namelist[i]}'s ({rolelist[i]}) Output:", ln=True, align='L')
        pdf.multi_cell(0, 10, tasklist[i].output)

    # Save the PDF to a file
    pdf_output = "consulting_tool_output.pdf"
    pdf.output(pdf_output)

    # Provide the PDF for download
    with open(pdf_output, "rb") as file:
        st.download_button("Download Results as PDF", file, file_name=pdf_output)
else:
    st.write('⬆️Ready to start')

# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")
