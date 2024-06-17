# agents.py
from crewai import Agent
from streamlitHelpers import create_streamlit_callback
from tools import * 
agent_emojis = {
    "Problem_Definition_Agent": "🔍",
    "Data_Assessment_Agent": "📊",
    "Model_Recommendation_Agent": "🤖",
    "Researcher": "📚",
    "Machine_Learning_Engineer": "💻",
    "Summarization_Agent": "📝"
}

def initialize_agents(llm):
    Problem_Definition_Agent = Agent(
        role='Problem_Definition_Agent',
        goal="""Clarify the machine learning problem the user wants to solve, identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
        backstory="""You are an expert in understanding and defining machine learning problems. Your goal is to extract a clear, concise problem statement from the user's input, ensuring the project starts with a solid foundation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        step_callback=create_streamlit_callback('Problem_Definition_Agent', agent_emojis['Problem_Definition_Agent'])
    )

    Data_Assessment_Agent = Agent(
        role='Data_Assessment_Agent',
        goal="""Evaluate the data provided by the user, assessing its quality, suitability for the problem, and analysing the statistical relationships between different variables suggesting preprocessing steps if necessary.""",
        backstory="""You specialize in data statistical evaluation and preprocessing. Your task is to guide the user in preparing their dataset for the machine learning model, including suggestions for data cleaning and augmentation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        step_callback=create_streamlit_callback('Data_Assessment_Agent', agent_emojis['Data_Assessment_Agent'])
    )



    Model_Recommendation_Agent = Agent(
        role='Model_Recommendation_Agent',
        goal="""Recommend the most suitable machine learning models based on the problem definition and data assessment.""",
        backstory="""You are an expert in machine learning model selection, capable of evaluating various models' strengths and weaknesses to provide the best recommendation for a given problem.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        # Add Arxiv Tool tools = 
        tools = [search_arxiv , perform_web_search],
        step_callback=create_streamlit_callback('Model_Recommendation_Agent', agent_emojis['Model_Recommendation_Agent'])
    )

    Researcher = Agent(
        role='Researcher',
        goal="""Perform in-depth research to gather necessary documentation, academic papers, and other resources relevant to the project.""",
        backstory="""You are a seasoned researcher, adept at finding and synthesizing information from a wide range of sources to support the team's objectives.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools = [perform_web_search , search_arxiv], 
        step_callback=create_streamlit_callback('Researcher', agent_emojis['Researcher'])
    )

    Machine_Learning_Engineer = Agent(
        role='Machine_Learning_Engineer',
        goal="""Generate starter Python code for the project, including data loading, model definition, and a basic training loop, based on findings from the problem definitions and data assessment.""",
        backstory="""You are a code wizard, able to generate starter code templates that users can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        step_callback=create_streamlit_callback('Machine_Learning_Engineer', agent_emojis['Machine_Learning_Engineer'])
    )

    Summarization_Agent = Agent(
        role='Summarization_Agent',
        goal="""Summarize complex information and findings into concise and understandable formats for the team.""",
        backstory="""You specialize in distilling large volumes of information into clear and actionable summaries, helping the team stay focused on key insights.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        step_callback=create_streamlit_callback('Summarization_Agent', agent_emojis['Summarization_Agent'])
    )

    return {
        "Problem_Definition": Problem_Definition_Agent,
        "Data_Assessment": Data_Assessment_Agent,
        "Model_Recommendation": Model_Recommendation_Agent,
        "Researcher": Researcher,
        "Machine_Learning_Engineer": Machine_Learning_Engineer,
        "Summarization": Summarization_Agent
    }
