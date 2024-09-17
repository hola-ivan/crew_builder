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
import random
from fpdf import FPDF, HTMLMixin
import markdown2

# List of examples
examples = [
    {"Business": "An online bakery specializing in custom cakes.", "Challenges": "Scaling delivery operations and optimizing digital marketing.", "Budget": "$2,000"},
    {"Business": "A small yoga studio offering virtual and in-person classes.", "Challenges": "Expanding customer base and implementing a booking system.", "Budget": "$1,500"},
    {"Business": "A solopreneur running a podcast on financial literacy.", "Challenges": "Monetizing content and growing audience engagement.", "Budget": "$500"},
    {"Business": "A boutique fashion brand for eco-friendly clothing.", "Challenges": "Sourcing sustainable materials and creating an e-commerce website.", "Budget": "$7,000"},
    {"Business": "A mobile app developer building fitness tracking software.", "Challenges": "Improving user experience and securing funding.", "Budget": "$10,000"},
    {"Business": "A local coffee shop that roasts its own beans.", "Challenges": "Increasing foot traffic and launching an online store.", "Budget": "$3,500"},
    {"Business": "A digital marketing agency offering social media management for small businesses.", "Challenges": "Client acquisition and improving automation in workflow.", "Budget": "$6,000"},
    {"Business": "A graphic designer focused on branding for startups.", "Challenges": "Establishing a niche and building a portfolio website.", "Budget": "$1,000"},
    {"Business": "A health and wellness coach offering personalized nutrition plans.", "Challenges": "Creating an online presence and setting up automated payment systems.", "Budget": "$800"},
    {"Business": "A local handyman service offering home repairs.", "Challenges": "Managing scheduling and integrating payment systems.", "Budget": "$4,000"}
]

# Randomly select three examples
selected_examples = random.sample(examples, 3)

# Load API key from secrets
serper_api_key = st.secrets["SERPER_API_KEY"]
os.environ["SERPER_API_KEY"] = serper_api_key  # serper.dev API key
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

# Existing sidebar code with added examples
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

    st.header("Example Scenarios")
    for i, example in enumerate(selected_examples, 1):
        st.subheader(f"Example {i}")
        st.write(f"**Business:** {example['Business']}")
        st.write(f"**Challenges:** {example['Challenges']}")
        st.write(f"**Budget:** {example['Budget']}")
        if st.button(f"Use Example {i}"):
            st.session_state['business_context'] = f"Business: {example['Business']}\nChallenges: {example['Challenges']}\nBudget: {example['Budget']}"

# Title for the app
st.title("effiweb solutions Consulting Tool")

# API Key setup
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)
# GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")
# GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# Input: Business context and challenges
business_context = st.text_area('Please tell us about **your Business, your Challenges, and your Budget**:', 
                                value=st.session_state.get('business_context', ''), 
                                help='The more details you provide, the better the agents can assist you.')
# Set the number of agents dynamically based on user input
# number_of_agents = st.slider('Number of Agents', min_value=2, max_value=10, value=4)
number_of_agents = 3

# Agent Setup
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To craft an efficient digital strategy tailored for your business context, optimizing growth, customer engagement, and scaling opportunities based on the following context: {}.",
        "task": "Design a tailored digital strategy for the provided business context, focusing on critical aspects such as online presence, scaling, and operational efficiency. Specific challenges provided by the client: {}.",
        "output": """
         1. Executive Summary
         2. Business Context and Challenges
         3. Key Initiatives
         4. Actionable Steps
         5. Budget Allocation
         6. Expected Outcomes
        """,
        "backstory": "Tlaloc specializes in digital strategies, helping small businesses build sustainable growth models in a dynamic digital landscape. Your expertise shines in identifying online marketing solutions, increasing customer engagement, and scaling businesses effectively."
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Specialist",
        "goal": "To integrate analytics and automation into the business model, ensuring data-driven decision-making and optimized customer engagement, focusing on the client’s specific challenges: {}.",
        "task": "Implement data-driven solutions for customer engagement and operational automation. Challenges and context provided by the client: {}.",
        "output": """
         1. Analytics Overview
         2. Automation Recommendations
         3. Key Metrics to Track
         4. Integration Plan
         5. Budget Overview
         6. Expected Outcomes
        """,
        "backstory": "Arminius is an expert in data analytics and automation, helping businesses streamline their operations and optimize customer engagement through cutting-edge technologies."
    },
    {
        "name": "Thusnelda",
        "role": "Marketing & Growth Specialist",
        "goal": "To develop a high-impact digital marketing strategy to enhance growth, customer acquisition, and brand visibility, based on the business context provided: {}.",
        "task": "Create a scalable digital marketing strategy with key focus areas such as SEO, social media engagement, and customer outreach. Client challenges and context: {}.",
        "output": """
         1. Marketing Strategy Overview
         2. Focus on Growth Opportunities
         3. Campaign Execution Plan
         4. Budget and ROI Estimates
         5. Key Growth Projections
        """,
        "backstory": "Thusnelda has expertise in helping startups and small businesses scale through targeted digital marketing strategies that drive results across social media, SEO, and customer engagement."
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

def check_for_additional_input(agent, task_description):
    # Check if agent needs more details to complete the task
    if "{extra_info}" in task_description:  # Example placeholder for missing info
        st.warning(f"{agent.role} needs more information to proceed.")
        additional_input = st.text_area(f"Please provide more details for {agent.role}:", "")
        if additional_input:
            # Replace the placeholder with the provided input
            task_description = task_description.replace("{extra_info}", additional_input)
            return task_description
        else:
            st.write("Waiting for input...")
            return None  # Halt task execution until input is provided
    return task_description

# Create click button
if st.button('Start'):
    agentlist, tasklist_full = [], []
    
    for i in range(number_of_agents):
        agent = Agent(
            role=rolelist[i],
            goal=goallist[i],
            backstory=backstorylist[i],
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=True,
            max_iter=4,
            memory=True
        )
        agentlist.append(agent)
        
        # Check for additional input if required by the task
        task_description = taskdescriptionlist[i]
        task_description = check_for_additional_input(agent, task_description)
        if task_description:  # Only proceed if the task description is complete
            tasklist_full.append(Task(description=task_description, expected_output=outputlist[i], agent=agent))
    
    if tasklist_full:  # Proceed if all tasks have sufficient information
        crew = Crew(agents=agentlist, tasks=tasklist_full, verbose=True, process=Process.sequential, full_output=True)
        results = crew.kickoff()

    # Generate PDF report with header, footer, and professional formatting
    class PDF(FPDF, HTMLMixin):
        def header(self):
            self.set_font("Arial", 'B', 14)
            self.cell(0, 10, "effiweb Solutions Consulting Report", 0, 1, 'C', link="https://effiweb.solutions")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Convert Markdown to HTML
    def markdown_to_html(markdown_content):
        return markdown2.markdown(markdown_content)

    # Generate PDF
    pdf = PDF()
    pdf.add_page()

    def add_multiline_text(pdf, title, text):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, title, 0, 1, 'L')
        pdf.ln(5)
        html_content = markdown_to_html(text)
        pdf.write_html(html_content)

    # Example usage
    for i in range(number_of_agents):
        add_multiline_text(pdf, f"Agent {i+1} - {updated_data[i]['role']}", "")
        add_multiline_text(pdf, "Proposed Solution", tasklist_full[i].output.raw)

    # Save the PDF
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