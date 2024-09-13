# Import necessary libraries for the application
from groq import Groq
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import json
import requests
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import load_tools
from crewai_tools import tool
from crewai import Crew, Process
import tomli  # Changed from tomllib to tomli
from langchain_groq import ChatGroq

# Set page configuration
st.set_page_config(page_title="Autonomous Crew Builder", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    /* Main background in a light shade of white */
    .main { 
        background-color: #f5f7fa; /* light white-blue */
    }

    /* Button styling with modern indigo */
    .stButton>button { 
        background-color: #4f46e5; /* indigo */
        color: #ffffff; /* white */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }

    /* Text input field styling with dark blue text */
    .stTextInput>div>div>input { 
        color: #1e3a8a; /* dark blue */
        background-color: #e0f2fe; /* light blue */
        border-radius: 5px;
        padding: 5px;
    }

    /* Label styling for input fields */
    .stTextInput>div>div>label { 
        color: #3b82f6; /* medium blue */
        font-weight: bold;
    }

    /* Textarea styling */
    .stTextArea>div>div>textarea { 
        color: #1e3a8a; /* dark blue */
        background-color: #e0f2fe; /* light blue */
        border-radius: 5px;
        padding: 5px;
    }

    /* Label styling for text areas */
    .stTextArea>div>div>label { 
        color: #3b82f6; /* medium blue */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Define collapsible sidebar sections
with st.sidebar:
    with st.expander("Contact", expanded=True):
        st.write("For any inquiries, contact me at [blog@holaivan.tech](mailto:blog@holaivan.tech)")

    with st.expander("How to use", expanded=False):
        st.write("This app allows you to create an autonomous crew of agents that can work together to achieve a common goal. Define the number of agents, assign them roles, goals, backstories, tasks, and expected outputs. The agents will work sequentially to achieve the common goal, and the app will display the output of each agent.")
        groq_api_key = st.text_input("Enter your Groq API key", type="password", help="You can get your API key from https://console.groq.com/keys")

    with st.expander("About", expanded=False):
        st.write("This tool helps create autonomous crews using AI agents. The agents work together to achieve specified goals by performing assigned tasks sequentially.")

    with st.expander("FAQ", expanded=False):
        st.write("Frequently Asked Questions about the Autonomous Crew Builder tool.")

# Create a title for the Streamlit app
st.title('Autonomous Crew Builder')

# Fetch the API key from the input box or secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize the Groq client with the API key
client = Groq(api_key=groq_api_key)

# Initialize the Groq Language Model with the API key and model details
GROQ_LLM = ChatGroq(
    api_key=groq_api_key,  # Ensure the API key is correctly passed here
    model="llama-3.1-70b-versatile"
)

# Ask the user to enter the number of agents that should be part of the crew
# number_of_agents = st.number_input('Enter the number of agents that should be part of the crew', min_value=1, max_value=10, value=3)
number_of_agents = 4

# Example data for pre-filling the form
example_data = [
    {
        "name": "Tlaloc",
        "role": "Digital Strategy Consultant",
        "goal": "To help solopreneurs and small businesses create effective digital strategies that align with their business goals.",
        "backstory": "An expert in working with small businesses, Aiden focuses on developing cost-effective digital roadmaps that can scale with growth.",
        "task": "Design a digital strategy for an early-stage startup, focusing on key priorities like online presence, customer acquisition, and scaling operations.",
        "output": "A digital strategy plan with key initiatives in website development, social media marketing, and customer management, including budget considerations."
    },
    {
        "name": "Tonantzin",
        "role": "Cloud Infrastructure Specialist",
        "goal": "To help startups and small businesses leverage cloud technology to streamline operations and reduce costs.",
        "backstory": "Specializing in affordable cloud solutions, Sofia helps clients implement cloud systems that grow with their business needs.",
        "task": "Set up a cloud infrastructure for a solopreneur’s e-commerce business, ensuring scalability and low operational costs.",
        "output": "A cloud architecture with clear implementation steps, cost projections, and automation features tailored to small business budgets."
    },
    {
        "name": "Arminius",
        "role": "Analytics & Automation Consultant",
        "goal": "To help solopreneurs and startups leverage data and automation to optimize their operations and improve customer engagement.",
        "backstory": "With a background in startups, Liam focuses on affordable analytics and automation tools that help small businesses track growth and customer behavior.",
        "task": "Implement an automated analytics system to track customer engagement and sales metrics for a small online business.",
        "output": "A dashboard providing real-time data on key performance indicators (KPIs), integrated with automated email and CRM workflows."
    },
    {
        "name": "Thusnelda",
        "role": "Digital Marketing & Growth Consultant",
        "goal": "To help small businesses and solopreneurs scale their operations through effective digital marketing strategies.",
        "backstory": "A digital marketing expert, Olivia has worked with early-stage startups to build brand awareness, acquire customers, and drive growth.",
        "task": "Develop and execute a digital marketing strategy, including SEO, social media, and email campaigns, for a small business looking to expand its customer base.",
        "output": "A comprehensive digital marketing plan with detailed timelines, budget allocations, and projected ROI, focusing on lead generation and brand growth."
    }
]


# Initialize lists to store the details of each agent
namelist = []
rolelist = []
goallist = []
backstorylist = []
taskdescriptionlist = []
outputlist = []
toollist = []

# Collect details for each agent using collapsible sections and pre-fill with example data
for i in range(0, number_of_agents):
    example = example_data[i] if i < len(example_data) else {"name": "", "role": "", "goal": "", "backstory": "", "task": "", "output": ""}
    with st.expander(f"Agent {i+1} Details", expanded=True):
        agent_name = st.text_input(f"Enter the name of agent {i+1}", value=example["name"])
        namelist.append(agent_name)
        role = st.text_input(f"Enter the role of agent {agent_name}", value=example["role"])
        rolelist.append(role)
        goal = st.text_input(f"Enter the goal of agent {agent_name}", value=example["goal"])
        goallist.append(goal)
        backstory = st.text_input(f"Describe the backstory of agent {agent_name}", value=example["backstory"])
        backstorylist.append(backstory)
        taskdescription = st.text_input(f"Describe the task of agent {agent_name}", value=example["task"])
        taskdescriptionlist.append(taskdescription)
        output = st.text_input(f"Describe the expected output of agent {agent_name}", value=example["output"])
        outputlist.append(output)

# Create Crew and display results
if st.button('Start'):
    agentlist = []
    tasklist = []
    for i in range(number_of_agents):
        agent = Agent(
            role=rolelist[i],
            goal=goallist[i],
            backstory=backstorylist[i],
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True
        )
        agentlist.append(agent)
        task = Task(
            description=taskdescriptionlist[i],
            expected_output=outputlist[i],
            agent=agent
        )
        tasklist.append(task)
    crew = Crew(
        agents=agentlist,
        tasks=tasklist,
        verbose=2,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work
    results = crew.kickoff()

    # Collect actual outputs
    actual_outputs = [task.output for task in tasklist]

    # Prepare the input for Groq API for summary generation
    summary_content = "Here are the outputs generated by the crew of agents:\n\n"
    for i, output in enumerate(actual_outputs):
        summary_content += f"{namelist[i]}: {output}\n"

    summary_content += "\nProvide a one line summary of the results for each agent and a crisp one line summary for the crew."

    summary_input = {
        "role": "user",
        "content": summary_content
    }

    # Request summary from Groq API
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[summary_input],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        stream=False
    )

    # Print response to debug the structure
    print("Response type:", type(completion))
    print("Response content:", completion)

    # Extract and display the summary text
    summary_text = completion.choices[0].message.content

    # Display results
    st.markdown("---")
    st.subheader("Summary")
        
    # Display the summary generated by Groq
    st.markdown(summary_text)

    # Display Actual Output
    st.markdown("---")
    st.subheader("Results")
    for i in range(0, number_of_agents):
        st.markdown(f"## {namelist[i]}'s Output")
        st.write(f"{tasklist[i].output}")
else:
    st.write('Please click the "Start" button to see the results')

# Add a call to action to contact
st.markdown("---")
st.write("If you have any questions or need assistance, feel free to contact me at [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech).")