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
        .main { background-color: #f4ebd0; }
        .stButton>button { background-color: #ffcc53; color: #122620; }
        .stTextInput>div>div>input { color: #122620; }
        .stTextInput>div>div>label { color: #122620; }
        .stTextArea>div>div>textarea { color: #122620; }
        .stTextArea>div>div>label { color: #122620; }
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
if 'GROQ_API_KEY' in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.session_state.get('groq_api_key', '')

# Initialize the Groq client with the API key
client = Groq(api_key=groq_api_key)

# Initialize the Groq Language Model with the API key and model details
GROQ_LLM = ChatGroq(
    api_key=groq_api_key,  # Ensure the API key is correctly passed here
    model="llama-3.1-70b-versatile"
)

# Ask the user to enter the number of agents that should be part of the crew
number_of_agents = st.number_input('Enter the number of agents that should be part of the crew', min_value=1, max_value=10, value=3)

# Example data for pre-filling the form
example_data = [
    {
        "name": "Robin",
        "role": "Content Researcher",
        "goal": "To gather the latest trends and insights in marketing research to include in the newsletter.",
        "backstory": "An expert in marketing analytics with a knack for finding emerging trends and valuable data.",
        "task": "Research and compile a list of recent studies, reports, and articles related to marketing research.",
        "output": "A comprehensive list of at least 10 recent sources, with summaries highlighting key insights."
    },
    {
        "name": "Maria",
        "role": "Newsletter Writer",
        "goal": "To create engaging and informative content for the newsletter based on the research provided.",
        "backstory": "A skilled writer with experience in crafting compelling newsletters and a deep understanding of marketing.",
        "task": "Write the newsletter, including an introduction, key findings from the research, and actionable recommendations for readers.",
        "output": "A polished draft of the newsletter, ready for review, including a catchy introduction, detailed sections on research findings, and a conclusion with actionable insights."
    },
    {
        "name": "Dana",
        "role": "Editor and Scheduler",
        "goal": "To ensure the newsletter is polished and scheduled for distribution.",
        "backstory": "An experienced editor with a keen eye for detail and a strong background in project management.",
        "task": "Review the newsletter draft for clarity, coherence, and correctness. Schedule the newsletter for distribution to the mailing list.",
        "output": "A final version of the newsletter with no errors, and a scheduled distribution date and time."
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
if st.button('Create Crew'):
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
        max_tokens=1024,
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
    st.write('Please click the button to create the crew and see the results')

# Add a call to action to contact
st.markdown("---")
st.write("If you have any questions or need assistance, feel free to contact me at [blog@holaivan.tech](mailto:blog@holaivan.tech).")