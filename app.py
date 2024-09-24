import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
import os
import random
from typing import List, Dict, Tuple
import time
from litellm import completion as litellm_completion


# Load API key from secrets
serper_api_key = st.secrets["SERPER_API_KEY"]
os.environ["SERPER_API_KEY"] = serper_api_key

# Set page configuration
st.set_page_config(page_title="effiweb solutions Consulting Tool", layout="wide")

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

# Caching examples
@st.cache_data
def load_examples() -> List[Dict[str, str]]:
    return [
        {
            "Business": "Sweet Creations: An online bakery specializing in custom cakes. Current Status: Established with moderate online presence.",
            "Challenges": "Limited delivery operations and basic digital marketing. Expected outcome after the project: Scale delivery operations and optimize digital marketing.",
            "Budget": "€2,000"
        },
        {
            "Business": "Tranquil Yoga: A small yoga studio offering virtual and in-person classes. Current Status: Small local following with some virtual classes.",
            "Challenges": "Limited customer base and manual booking system. Expected outcome after the project: Expand customer base and implement an automated booking system.",
            "Budget": "$1,500"
        },
        {
            "Business": "Money Talks: A solopreneur running a podcast on financial literacy. Current Status: Growing audience with limited monetization.",
            "Challenges": "Limited monetization and audience engagement. Expected outcome after the project: Monetize content and grow audience engagement.",
            "Budget": "MXN 10,000"
        },
    ]

# Sidebar setup
def setup_sidebar():
    with st.sidebar:
        st.header("Example Scenarios")
        examples = load_examples()
        for i, example in enumerate(examples, 1):
            with st.expander(f"Example {i}", expanded=False):
                st.write(f"**Business:** {example['Business']}")
                st.write(f"**Challenges:** {example['Challenges']}")
                st.write(f"**Budget:** {example['Budget']}")
                if st.button(f"Use Example {i}", key=f"use_example_{i}"):
                    st.session_state['business_context'] = f"Business: {example['Business']}\nChallenges: {example['Challenges']}\nBudget: {example['Budget']}"

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

            **Q: How should I interpret the results?**
            A: The results provide strategic recommendations based on AI analysis. Use them as a starting point for your business strategy, but always combine with your industry knowledge and specific context.

            **Q: How can I implement the suggested strategies?**
            A: Start by prioritizing the recommendations based on your immediate needs and resources. Create an action plan with specific timelines and responsible team members for each task.
            """)

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    return Groq(api_key=groq_api_key), ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# Input validation
def validate_input(business_context: str) -> bool:
    if not business_context.strip():
        st.error("Please provide information about your business and challenges.")
        return False
    if len(business_context.split()) < 20:
        st.warning("Please provide more detailed information for better results.")
        return False
    return True

# Update agent goals and tasks
def update_agent_goals_and_tasks(context: str, agent_data: List[Dict]) -> List[Dict]:
    updated_agents = []
    for agent in agent_data:
        agent["goal"] = agent["goal"].format(context)
        agent["task"] = agent["task"].format(context, context, extra_info="")
        updated_agents.append(agent)
    return updated_agents

# Human-in-the-loop for task execution
def check_for_additional_input(agent: Dict, task_description: str, task_output: str) -> Tuple[str, str]:
    if "{extra_info}" in task_description or "{extra_info}" in task_output:
        st.warning(f"{agent['role']} needs more information to proceed.")
        additional_input = st.text_area(f"Provide extra details for {agent['role']} to proceed:", key=f"input_{agent['role']}")
        if additional_input:
            task_description = task_description.format(extra_info=additional_input)
            task_output = task_output.format(extra_info=additional_input)
            st.success(f"Input received! {agent['role']} will continue.")
            return task_description, task_output
        else:
            st.write(f"Waiting for input from the user for {agent['role']}...")
            return None, None
    return task_description, task_output

# Main function to run the consulting process
def run_consulting_process(business_context: str, agent_data: List[Dict], client: Groq, llm: ChatGroq):
    updated_agents = update_agent_goals_and_tasks(business_context, agent_data)
    agentlist, tasklist = [], []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, agent_data in enumerate(updated_agents):
        status_text.text(f"Processing agent {i+1}/{len(updated_agents)}: {agent_data['role']}")
        
        agent = Agent(
            role=agent_data['role'],
            goal=agent_data['goal'],
            backstory="",
            llm=llm,
            verbose=False,
            allow_delegation=True,
            max_iter=6,
            memory=True
        )
        
        task_description, task_output = check_for_additional_input(agent_data, agent_data['task'], agent_data['output'])

        if task_description and task_output:
            task = Task(description=task_description, expected_output=task_output, agent=agent)
            agentlist.append(agent)
            tasklist.append(task)

        progress_bar.progress((i + 1) / len(updated_agents))
        time.sleep(0.5)  # Simulate processing time

    if tasklist:
        status_text.text("Running the crew...")
        crew = Crew(agents=agentlist, tasks=tasklist, process=Process.sequential, full_output=True)
        try:
            results = crew.kickoff(inputs={'business_context': business_context})
            return results
        except Exception as e:
            st.error(f"An error occurred while processing: {str(e)}")
            return None
    else:
        st.warning("No tasks were created. Please check your input and try again.")
        return None

# Display results in a more structured format
def display_results(results: List[str]):
    if results:
        for i, result in enumerate(results, 1):
            with st.expander(f"Strategy {i}: {result.split('\n')[0]}", expanded=False):
                st.markdown(result)
    else:
        st.warning("No results were generated. Please try again with more detailed input.")

# Main app
def main():
    st.title("effiweb solutions Consulting Tool")
    setup_sidebar()

    business_context = st.text_area('Please tell us about **your Business, your Challenges, and your Budget**:', 
                                    value=st.session_state.get('business_context', ''), 
                                    help='The more details you provide, the better the agents can assist you.')

    if st.button('Start Consulting Process'):
        if validate_input(business_context):
            with st.spinner('Processing your request...'):
                client, llm = init_groq_client()
                results = run_consulting_process(business_context, example_data, client, llm)
                if results:
                    display_results(results)

    st.markdown("---")
    st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")

if __name__ == "__main__":
    main()