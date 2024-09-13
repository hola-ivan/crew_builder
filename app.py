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
import pandas as pd
import csv

# Set page configuration
st.set_page_config(page_title="effiweb solutions Consulting Tool", layout="wide")

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
        st.write("For any inquiries, contact me at [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")

    with st.expander("About", expanded=False):
        st.write("""
        The effiweb solutions **Consulting Tool** helps create autonomous crews using AI agents. These agents work together to achieve specified goals by performing assigned tasks sequentially.

        **Disclaimer**: This tool is for educational purposes only and is not intended to replace professional consulting or business services. Always consult with a professional for tailored advice and solutions.
        """)

    with st.expander("FAQ", expanded=False):
        st.write("""
        **Q: What is the purpose of this tool?**  
        A: This tool is designed to provide digital solutions and strategies for solopreneurs, early-stage startups, and small businesses. It helps with creating effective digital strategies, implementing cloud infrastructure, and utilizing data and automation.

        **Q: Is this tool a replacement for professional consulting?**  
        A: No, this tool is for educational purposes only. It offers general advice and recommendations but should not replace professional consulting or business services. For tailored advice, consult with a professional.

        **Q: How can I use this tool effectively?**  
        A: To use this tool effectively, follow the provided guidelines for digital strategy, cloud infrastructure, analytics, and digital marketing. Adjust the recommendations based on your specific business needs and goals.

        **Q: How do I get help if I encounter issues?**  
        A: For assistance, please refer to the [Contact](mailto:effiwebsolutions@holaivan.tech) section.

        **Q: Are there any costs associated with using this tool?**  
        A: The tool itself is provided for educational purposes and does not have a cost. However, implementing recommendations may involve costs depending on your chosen solutions and services.
        """)

    with st.expander("Acknowledgements", expanded=False):
        st.write("""
        This tool is based on the work of the original repository licensed under the MIT License. 

        **Original Project**: [DriesFaems/crew_builder](https://github.com/DriesFaems/crew_builder)

        **MIT License**:
        ```
        MIT License

        Copyright (c) 2024 Ivan Escamilla Rodriguez

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
        THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        ```
        """)

# Create a title for the Streamlit app
st.title('effiweb solutions Consulting Tool')

groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize the Groq client with the API key
client = Groq(api_key=groq_api_key)

# Initialize the Groq Language Model with the API key and model details
GROQ_LLM = ChatGroq(
    api_key=groq_api_key,  # Ensure the API key is correctly passed here
    model="llama-3.1-70b-versatile"
)

# Ask the user to enter the number of agents that should be part of the crew
number_of_agents = st.number_input('Enter the number of agents that should be part of the crew', min_value=1, max_value=10, value=3)

def read_csv_robust(file_path):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            data = list(reader)
        
        # Find the maximum number of columns
        max_columns = max(len(row) for row in data)
        
        # Pad rows with fewer columns
        padded_data = [row + [''] * (max_columns - len(row)) for row in data]
        
        # Use the first row as column names, replacing empty strings with placeholder names
        columns = [col if col else f'Column_{i}' for i, col in enumerate(padded_data[0])]
        
        # Create DataFrame, skipping the header row
        df = pd.DataFrame(padded_data[1:], columns=columns)
        
        # Convert 'Number of Agents' to numeric, coercing errors to NaN
        df['Number of Agents'] = pd.to_numeric(df['Number of Agents'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Use the robust method to read the CSV
df = read_csv_robust('examples.csv')

# Print the column names and first few rows for debugging
#st.write("Column names:", df.columns.tolist())
#st.write("\nFirst few rows:")
#st.write(df.head())

# Filter the dataframe for rows with the selected number of agents
filtered_df = df[df['Number of Agents'] == number_of_agents]

# If there are no examples for the selected number of agents, choose the closest available
if filtered_df.empty:
    closest_number = df['Number of Agents'].dropna().iloc[(df['Number of Agents'].dropna() - number_of_agents).abs().argsort()[:1]].values[0]
    filtered_df = df[df['Number of Agents'] == closest_number]
    st.warning(f"No example found for {number_of_agents} agents. Using an example with {closest_number} agents instead.")

# Randomly select one row from the filtered dataframe
if not filtered_df.empty:
    selected_row = filtered_df.sample(n=1).iloc[0]
    
    # Create the example_data list
    example_data = []
    for i in range(1, int(selected_row['Number of Agents']) + 1):
        agent_data = {
            "name": f"Agent {i}",
            "role": selected_row.get(f'Agent {i} Role', f"Role {i}"),
            "goal": f"To contribute to the {selected_row['Use Case']} project effectively.",
            "backstory": f"An experienced professional in {selected_row.get(f'Agent {i} Role', f'Role {i}')} with a strong background in the field.",
            "task": f"Perform duties related to {selected_row.get(f'Agent {i} Role', f'Role {i}')} in the context of {selected_row['Use Case']}.",
            "output": f"A comprehensive report or analysis related to {selected_row.get(f'Agent {i} Role', f'Role {i}')} for the {selected_row['Use Case']} project."
        }
        example_data.append(agent_data)

    # Display the selected use case
    st.write(f"## Example Use Case: {selected_row['Use Case']}")
else:
    st.error("No suitable examples found in the CSV file.")

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
    with st.expander(f"Agent {i+1} Details", expanded=False):
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
        summary_content += f"{namelist[i]}({rolelist[i]}): {output}\n"

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
    #print("Response type:", type(completion))
    #print("Response content:", completion)

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
        st.markdown(f"## {namelist[i]}'s ({rolelist[i]}) Output")
        st.write(f"{tasklist[i].output}")
else:
    st.write('Please click the button to create the crew and see the results')

# Add a call to action to contact
st.markdown("---")
st.write("If you have any questions or need assistance, feel free to contact me at [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech).")