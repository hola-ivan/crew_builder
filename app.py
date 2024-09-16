# Import necessary libraries
from groq import Groq
import streamlit as st
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
import pandas as pd
import csv
import os
from fpdf import FPDF
import time
import threading


os.environ["SERPER_API_KEY"] = "b24b3c1b64b69f8af2616fa008942e04f5df6f28" # serper.dev API key
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Set page configuration
st.set_page_config(page_title="effiweb solutions Consulting Tool", layout="wide")

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Arial:wght@400;700&display=swap');
        body, input, button, textarea { font-family: 'Arial', sans-serif; }
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
# GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")
# GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# Input: Business context and challenges
business_context = st.text_area('Plase tell us about **your Business, your Challenges, and your Budget**:', help='The more details you provide, the better the agents can assist you.')

# Set the number of agents dynamically based on user input
# number_of_agents = st.slider('Number of Agents', min_value=2, max_value=10, value=4)
number_of_agents = 3


# Example data for agents
example_data = [

    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To help solopreneurs and small businesses create effective digital strategies that align with their business goals, considering the context provided by the client: {}.",
        "task": "Design a digital strategy for a business, focusing on key priorities like online presence, customer acquisition, and scaling operations. context provided by the client: {}.",
        "output": "A digital strategy plan with key initiatives in website development, social media marketing, and customer management, including budget considerations.",
        "backstory": "Tlaloc has over 10 years of experience in digital strategy, having worked with numerous startups and small businesses to enhance their online presence and growth. context provided by the client: {}.",
    },
    # {
    #     "name": "Tonantzin",
    #     "role": "Cloud Infrastructure Specialist",
    #     "goal": "To help businesses leverage cloud technology to streamline operations and reduce costs in line with the given context provided by the client: {}.",
    #     "task": "Set up a cloud infrastructure, ensuring scalability and low operational costs based on the business needs. context provided by the client: {}.",
    #     "output": "A cloud architecture with clear implementation steps, cost projections, and automation features tailored to small business budgets.",
    #     "backstory": "Tonantzin is a certified cloud architect with a background in helping businesses transition to cloud solutions, focusing on cost efficiency and scalability. context provided by the client: {}."
    # },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Consultant",
        "goal": "To help businesses leverage data and automation to optimize operations, focusing on customer engagement in the context provided by the client: {}.",
        "task": "Implement an automated analytics system to track customer engagement and sales metrics for the business. context provided by the client: {}.",
        "output": "A dashboard providing real-time data on key performance indicators (KPIs), integrated with automated email and CRM workflows.",
        "backstory": "Arminius has a strong background in data science and automation, having implemented analytics solutions for various industries to improve customer engagement and operational efficiency. context provided by the client: {}."
    },
    {
        "name": "Thusnelda",
        "role": "Digital Marketing & Growth Consultant",
        "goal": "To help businesses scale their operations through effective digital marketing strategies, with a focus on growth within the context provided by the client: {}.",
        "task": "Develop and execute a digital marketing strategy, including SEO, social media, and email campaigns for the business. context provided by the client: {}.",
        "output": "A comprehensive digital marketing plan with detailed timelines, budget allocations, and projected ROI, focusing on lead generation and brand growth.",
        "backstory": "Thusnelda is an expert in digital marketing with a proven track record of helping businesses grow their online presence and customer base through targeted marketing strategies. context provided by the client: {}."
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
    namelist.append(agent["name"])
    rolelist.append(agent["role"])
    goallist.append(agent["goal"])
    backstorylist.append(agent["backstory"])
    taskdescriptionlist.append(agent["task"])
    outputlist.append(agent["output"])

# Create click button
if st.button('Start'):
    agentlist, tasklist_full = [], []

    for i in range(number_of_agents):
        agent = Agent(
            role=rolelist[i],
            goal=goallist[i],
            backstory=backstorylist[i],  # Ensure backstory is provided
            llm=GROQ_LLM,
            tools=[search_tool,web_rag_tool],
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

    # Generate PDF report with header, footer, and professional formatting
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", 'B', 14)  # Larger, bold title
            self.cell(0, 10, "effiweb Solutions Consulting Report", 0, 1, 'C', link="https://effiweb.solutions")
            self.ln(5)  # Reduce space after the title
            self.set_font("Arial", 'I', 10)
            self.ln(10)  # Slight space before content

        def footer(self):
            self.set_y(-15)  # Position footer at bottom
            self.set_font("Arial", 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font("Arial", 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(5)

        def chapter_body(self, body):
            self.set_font("Arial", '', 12)
            self.multi_cell(0, 8, body)  # Reduce line height for more compact text
            self.ln(3)  # Slightly smaller gap between paragraphs

        def add_table(self, headers, data):
            # Basic table with headers and data
            self.set_font("Arial", 'B', 12)
            col_width = self.w / 6.5  # Adjust column width based on page width
            for header in headers:
                self.cell(col_width, 10, header, border=1, align='C')
            self.ln()

            self.set_font("Arial", '', 12)
            for row in data:
                for item in row:
                    self.cell(col_width, 10, str(item), border=1, align='C')
                self.ln()


# Create the PDF document
    pdf = PDF()
    pdf.add_page()

    def add_multiline_text(pdf, title, text):
        pdf.chapter_title(title)
        pdf.chapter_body(text)

    for i in range(number_of_agents):
        add_multiline_text(pdf, f"Agent {i+1} - {updated_data[i]['role']}", "")
        # add_multiline_text(pdf, "Role", updated_data[i]['role'])
        # add_multiline_text(pdf, "Goal", updated_data[i]['goal'])
        # add_multiline_text(pdf, "Task Description", updated_data[i]['task'])
        # add_multiline_text(pdf, "Expected Output", updated_data[i]['output'])
        add_multiline_text(pdf, "Proposed Solution", tasklist_full[i].output.raw)

    # Save PDF
    pdf_file = "effiweb.solutions_free_consulting_report.pdf"
    pdf.output(pdf_file)
    
    # Provide a download link
    with open(pdf_file, "rb") as f:
        st.download_button("Download Detailed Report", f, file_name=pdf_file)
else:
    st.write('Click "Start" to generate results.')

# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")