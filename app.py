import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task
import os
from typing import List, Dict

# Load API key from secrets
serper_api_key = st.secrets["SERPER_API_KEY"]
os.environ["SERPER_API_KEY"] = serper_api_key

# Set page configuration
st.set_page_config(page_title="effiweb solutions Consulting Tool", layout="wide")

# Example agent data
example_data = [
    {
        "name": "Tlaloc",
        "backstory": "Tlaloc is a seasoned Digital Strategy Consultant with a passion for crafting innovative solutions for businesses of all sizes. With a background in marketing and technology, Tlaloc has helped numerous clients achieve their growth objectives through strategic digital initiatives.",
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
    },
    {
        "name": "Arminius",
        "backstory": "Arminius is a seasoned Data Scientist with a knack for uncovering actionable insights from complex datasets. With a background in machine learning and analytics, Arminius has helped businesses leverage their data to drive strategic decision-making and operational efficiency.",
        "role": "Analytics & Automation Specialist",
        "goal": "To integrate analytics and automation into the business model, ensuring data-driven decision-making and optimized customer engagement, focusing on the client's specific challenges: {}.",
        "task": "Implement data-driven solutions for customer engagement and operational automation. Challenges and context provided by the client: {}.",
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
        "backstory": "Thusnelda is a seasoned UX/UI Designer with a passion for creating intuitive and engaging user experiences. With a background in design and human-computer interaction, Thusnelda has helped businesses enhance their digital products and services through user-centric design principles.",
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
     return Groq(api_key=groq_api_key), ChatGroq(api_key=groq_api_key, model="groq/llama-3.1-70b-versatile")

# Input validation
def validate_input(business_context: str) -> bool:
    if not business_context.strip():
        st.error("Please provide information about your business and challenges.")
        return False
    if len(business_context.split()) < 20:
        st.warning("Please provide more detailed information for better results.")
        return False
    return True

# Main function to run the consulting process
def run_consulting_process(business_context: str, agent_data: List[Dict], client: Groq, llm: ChatGroq):
    # Create agents and tasks based on the provided business context
    agents = []
    tasks = []
    for data in agent_data:
        agent = Agent(
            role=data['role'],
            goal=data['goal'].format(business_context),
            backstory=data['backstory'],
            verbose=False,
        )
        task = Task(
            description=data['task'].format(business_context),
            expected_output=data['output'],
            agent=agent,
        )
        agents.append(agent)
        tasks.append(task)

    my_crew = Crew(agents=agents, tasks=tasks)
    results = my_crew.kickoff(inputs={"input": business_context})
    return results

# Function to display results in a simpler format
def display_results(results: List[str]):
    if results:
        for i, result in enumerate(results, 1):
            st.subheader(f"Strategy {i}")
            st.markdown(result)
    else:
        st.warning("No results were generated. Please try again with more detailed input.")

# Main app logic
def main():
    st.title("effiweb solutions Consulting Tool")
    setup_sidebar()

    # Capture business context input from user
    business_context = st.text_area('Please tell us about **your Business, your Challenges, and your Budget**:', 
                                    value=st.session_state.get('business_context', ''), 
                                    help='The more details you provide, the better the agents can assist you.')

    # Trigger the consulting process on button click
    if st.button('Start Consulting Process'):
        if validate_input(business_context):  # Ensure the input is valid
            with st.spinner('Processing your request...'):
                client, llm = init_groq_client()  # Initialize Groq client and LLM
                results = run_consulting_process(business_context, example_data, client, llm)  # Run the process

                if results:
                    display_results(results)  # Display results in a structured format

    st.markdown("---")
    st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")

if __name__ == "__main__":
    main()