# effiweb solutions Consulting Tool

The effiweb solutions **Consulting Tool** helps build autonomous agent crews to simulate a consulting team. This tool is designed for educational purposes and not for professional consulting.

## Features

- Create a team of AI agents with specific roles and tasks.
- Customize agents' goals, tasks, and backstories based on business context.
- Generate detailed PDF reports of the consulting process.
- Interactive UI with real-time results.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/effiweb-solutions-consulting-tool.git
    cd effiweb-solutions-consulting-tool
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Application**:
    ```sh
    streamlit run app.py
    ```

2. **Set Up Your Crew**:
    - Enter the business context, challenges, and budget.
    - Specify the number of agents (default is 3).
    - Click the **Start** button to generate results.

3. **View Results**:
    - The agents will work together to provide solutions based on the provided context.
    - Download the detailed PDF report of the consulting process.

## Configuration

1. **API Keys**:
    - Add your API keys to the Streamlit secrets:
        ```sh
        streamlit secrets set SERPER_API_KEY "your_serper_api_key"
        streamlit secrets set GROQ_API_KEY "your_groq_api_key"
        ```

## Example

1. **Business Context**:
    - Provide details about your business, challenges, and budget.

2. **Agents**:
    - Example agents include:
        - **Tlaloc**: Digital Strategy Consultant
        - **Arminius**: Analytics & Automation Consultant
        - **Thusnelda**: Digital Marketing & Growth Consultant

3. **Output**:
    - The agents will generate a digital strategy plan, analytics dashboard, and marketing plan based on the provided context.

## Contact

Questions? Contact: [effiwebsolutions@holaivan.tech](mailto:effiwebsolutions@holaivan.tech)