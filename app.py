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
import os
import random

# List of examples
examples = [
    {
        "Business": "Sweet Creations: An online bakery specializing in custom cakes. Current Status: Established with moderate online presence.",
        "Challenges": "Limited delivery operations and basic digital marketing. Expected outcome after the project: Scale delivery operations and optimize digital marketing.",
        "Budget": "€2,000"
    },
    {
        "Business": "Tranquil Yoga: A small yoga studio offering virtual and in-person classes.  Current Status: Small local following with some virtual classes.",
        "Challenges": "Limited customer base and manual booking system. Expected outcome after the project: Expand customer base and implement an automated booking system.",
        "Budget": "$1,500"
    },
    {
        "Business": "Money Talks: A solopreneur running a podcast on financial literacy. Current Status: Growing audience with limited monetization.",
        "Challenges": "Limited monetization and audience engagement. Expected outcome after the project: Monetize content and grow audience engagement.",
        "Budget": "MXN 10,000"
    },
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

# Sidebar for examples and contact information
with st.sidebar:
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
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
# GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

# Input: Business context and challenges
business_context = st.text_area('Please tell us about **your Business, your Challenges, and your Budget**:', 
                                value=st.session_state.get('business_context', ''), 
                                help='The more details you provide, the better the agents can assist you.')

# Number of agents (static for simplicity)
number_of_agents = 3

# Example agent data
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To craft an efficient digital strategy tailored for your business context, optimizing growth, customer engagement, and scaling opportunities based on the following context: {}.",
        "task": "Design a tailored digital strategy for the provided business context, focusing on critical aspects such as online presence, scaling, and operational efficiency. Specific challenges provided by the client: {}. If additional information is needed, please specify: {extra_info}.",
        "output": """
         1. Executive Summary
         2. Business Context and Challenges
         3. Key Initiatives
         4. Actionable Steps
         5. Budget Allocation
         6. Expected Outcomes
        """,
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Specialist",
        "goal": "To integrate analytics and automation into the business model, ensuring data-driven decision-making and optimized customer engagement, focusing on the client's specific challenges: {}.",
        "task": "Implement data-driven solutions for customer engagement and operational automation. Challenges and context provided by the client: {}. If additional information is needed, please specify: {extra_info}.",
        "output": """
         1. Analytics Overview
         2. Automation Recommendations
         3. Key Metrics to Track
         4. Integration Plan
         5. Budget Overview
         6. Expected Outcomes
        """,
    },
    {
        "name": "Thusnelda",
        "role": "Marketing & Growth Specialist",
        "goal": "To develop a high-impact digital marketing strategy to enhance growth, customer acquisition, and brand visibility, based on the business context provided: {}.",
        "task": "Create a scalable digital marketing strategy with key focus areas such as SEO, social media engagement, and customer outreach. Client challenges and context: {}. If additional information is needed, please specify: {extra_info}.",
        "output": """
         1. Marketing Strategy Overview
         2. Focus on Growth Opportunities
         3. Campaign Execution Plan
         4. Budget and ROI Estimates
         5. Key Growth Projections
        """,
    }
]

# Update agent goals and tasks with user context
def update_agent_goals_and_tasks(context):
    updated_agents = []
    for agent_data in example_data:
        agent_data["goal"] = agent_data["goal"].format(context)
        agent_data["task"] = agent_data["task"].format(context, context, extra_info="")
        updated_agents.append(agent_data)
    return updated_agents

# Implement human-in-the-loop for task execution
def check_for_additional_input(agent, task_description, task_output):
    if "{extra_info}" in task_description or "{extra_info}" in task_output:
        st.warning(f"{agent.role} needs more information to proceed.")
        additional_input = st.text_area(f"Provide extra details for {agent.role} to proceed:", key=f"input_{agent.role}")
        if additional_input:
            task_description = task_description.format(extra_info=additional_input)
            task_output = task_output.format(extra_info=additional_input)
            st.success(f"Input received! {agent.role} will continue.")
            return task_description, task_output
        else:
            st.write(f"Waiting for input from the user for {agent.role}...")
            return None, None
    return task_description, task_output

# Start the consulting process
if st.button('Start'):
    if business_context:
        updated_agents = update_agent_goals_and_tasks(business_context)
        agentlist, tasklist = [], []

        for i in range(number_of_agents):
            agent_data = updated_agents[i]
            agent = Agent(
                role=agent_data['role'],
                goal=agent_data['goal'],
                backstory="",
                llm=GROQ_LLM,
                verbose=False,
                allow_delegation=True,
                max_iter=6,
                memory=True
            )
            
            task_description = agent_data['task']
            task_output = agent_data['output']
            task_description, task_output = check_for_additional_input(agent, task_description, task_output)

            if task_description and task_output:
                task = Task(description=task_description, expected_output=task_output, agent=agent)
                agentlist.append(agent)
                tasklist.append(task)
                if tasklist:
                    crew = Crew(agents=agentlist, tasks=tasklist, process=Process.sequential, full_output=True)
                    results = crew.kickoff(inputs={'business_context': business_context})
                    
                    # Display the result in markdown format with collapsible sections
                    for i, result in enumerate(results, 1):
                        st.markdown(f"### Result {i}")
                        st.markdown(f"<details><summary>Click to expand</summary><p>{result}</p></details>", unsafe_allow_html=True)