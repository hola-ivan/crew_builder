# Import necessary libraries
from groq import Groq
import streamlit as st
from langchain_groq import ChatGroq
from crewai import Crew, Agent, Task, Process
from fpdf import FPDF  # For PDF generation
import time

# Set page configuration
st.set_page_config(page_title="effiweb Solutions Consulting Tool", layout="wide")

# Custom CSS for modern look and feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        body, input, button, textarea { font-family: 'Inter', sans-serif; }
        .main { background-color: #f5f7fa; }
        .stButton>button { background-color: #4f46e5; color: #fff; border-radius: 5px; padding: 10px 20px; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { color: #1e3a8a; background-color: #e0f2fe; border-radius: 5px; padding: 5px; }
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
        A: Answer the interview questions to provide detailed business context. The agents will use this data to build strategies tailored to your needs.
        """)

# Title for the app
st.title("effiweb solutions Consulting Tool")

# API Key setup
groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)
GROQ_LLM = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile")

# --- User interview for business context ---
st.subheader("Business Context Interview")
business_name = st.text_input('What is your business name?', help='Enter the name of your business.')
business_goal = st.text_area('What are your business goals?', help='Describe your top business objectives.')
target_audience = st.text_area('Who is your target audience?', help='Describe your target customers or market.')
budget = st.number_input('What is your budget for this project?', min_value=100, max_value=100000, step=500)

# Collect input data into a context for the agents
if business_name and business_goal and target_audience and budget:
    business_context = f"Business Name: {business_name}\n" \
                       f"Goals: {business_goal}\n" \
                       f"Target Audience: {target_audience}\n" \
                       f"Budget: {budget}"
    st.success("Business context successfully captured!")
else:
    st.warning("Please complete all fields.")

# Set the number of agents dynamically based on user input
number_of_agents = st.slider('Number of Agents', min_value=2, max_value=10, value=4)

# Skeleton loader to simulate progress
def show_skeleton_loader(duration=3):
    st.text("Processing, please wait...")
    for _ in range(duration):
        st.progress(_ / duration)
        time.sleep(1)

# Example data for agents
example_data = [
    # Digital Strategy Consultant, Cloud Infrastructure Specialist, Analytics & Automation Consultant, Digital Marketing Consultant
    {
        "name": "Tlaloc", "role": "Digital Strategy Consultant",
        "goal": "To help solopreneurs and small businesses create effective digital strategies. Context: {}.",
        "task": "Design a digital strategy focused on online presence, customer acquisition, and scaling. Context: {}.",
        "output": "A digital strategy plan with key initiatives in website development and social media."
    },
    # Add other agents (Tonantzin, Arminius, Thusnelda) with similar structure...
]

# Function to update agent details
def update_agent_goals_and_tasks(context):
    updated_agents = []
    for agent_data in example_data:
        agent_data["goal"] = agent_data["goal"].format(context)
        agent_data["task"] = agent_data["task"].format(context)
        updated_agents.append(agent_data)
    return updated_agents

# Updating agents with context
if business_context:
    updated_data = update_agent_goals_and_tasks(business_context)

# Start the process
if st.button('Start'):
    agentlist, tasklist_full = [], []
    
    # Show skeleton loader while processing
    show_skeleton_loader()
    
    for i in range(number_of_agents):
        agent = Agent(
            role=updated_data[i]["role"],
            goal=updated_data[i]["goal"],
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=True,
            max_iter=5,
            memory=True
        )
        tasklist_full.append(Task(description=updated_data[i]["task"], expected_output=updated_data[i]["output"], agent=agent))
    
    crew = Crew(agents=agentlist, tasks=tasklist_full, verbose=True, process=Process.sequential, full_output=True)
    results = crew.kickoff()

    # Short summary of results
    st.write("**Consultation Summary**")
    st.write("We have completed the analysis. A detailed PDF report is available for download.")

    # Generating PDF with detailed results
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="effiweb Solutions Consulting Report", ln=True, align='C')

    for i in range(number_of_agents):
        pdf.cell(200, 10, f"Agent {i+1} - {updated_data[i]['name']}", ln=True)
        pdf.cell(200, 10, f"Role: {updated_data[i]['role']}", ln=True)
        pdf.cell(200, 10, f"Goal: {updated_data[i]['goal']}", ln=True)
        pdf.cell(200, 10, f"Task Description: {updated_data[i]['task']}", ln=True)
        pdf.cell(200, 10, f"Expected Output: {updated_data[i]['output']}", ln=True)

    # Save PDF
    pdf_file = "consulting_report.pdf"
    pdf.output(pdf_file)
    
    # Provide a download link
    with open(pdf_file, "rb") as f:
        st.download_button("Download Detailed Report", f, file_name=pdf_file)
else:
    st.write('Please complete the interview and click "Start" to generate results.')

# Contact section
st.markdown("---")
st.write("Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)")