# tasks.py
from typing import Dict, List, Any
import pandas as pd
from crewai import Agent, Task

def setup_tasks(
    agents: Dict[str, Agent],
    user_question: str,
    df: pd.DataFrame,
    uploaded_file: Any
    )  -> List[Task]:
    """
    Sets up tasks for agents to define the problem, assess data, recommend models, and generate code.

    Args:
        agents (Dict[str, Agent]): Dictionary of agents to handle different tasks.
        user_question (str): The user's machine learning problem statement.
        df (pd.DataFrame): The dataset provided by the user.
        uploaded_file (Any): The uploaded file object containing the dataset.

    Returns:
        List[Task]: A list of tasks to be executed by the agents.
    """

    task_define_problem = Task(
        description=f"Clarify and define the machine learning problem, including identifying the problem type and specific requirements. Here is the user's problem: {user_question}",
        agent=agents["Problem_Definition"],
        expected_output="A clear and concise definition of the machine learning problem."
    )

    task_assess_data = Task(
        description=f"Evaluate the user's data for quality and suitability and give the most detailed statistics of the data available. Here is a sample of the user's data: {df.head()} The file name is {uploaded_file.name}. Here is a statistical description of the data: {df.describe(include='all')}",
        agent=agents["Data_Assessment"],
        expected_output="An assessment of the data's metadata, quality and suitability, with suggestions for preprocessing or augmentation if necessary. The output must contain comparatif , qualitatif and quantitatif tables in markdown",
        context=[task_define_problem]
    )

    task_recommend_model = Task(
        description="Suggest suitable machine learning models for the defined problem and assessed data, providing rationale for each suggestion.",
        agent=agents["Model_Recommendation"],
        expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion.",
        context=[task_define_problem, task_assess_data]
    )

    task_research_docs = Task(
        description="Conduct research to gather necessary documentation, academic papers, and other resources relevant to the project.",
        agent=agents["Researcher"],
        expected_output="A summary of relevant research findings.",
        context=[task_recommend_model]
    )

    task_generate_code = Task(
        description="Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), including snippets for package import, data handling, model definition, and training.",
        agent=agents["Machine_Learning_Engineer"],
        expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project.",
        context=[task_define_problem, task_recommend_model, task_research_docs]
    )

    task_reflect_on_code = Task(
        description="Review and reflect on the generated Python code to ensure it meets the project requirements and is optimized for performance.",
        agent=agents["Machine_Learning_Engineer"],
        expected_output="Reflections and suggestions for improving the initial code.",
        context=[task_generate_code, task_define_problem, task_research_docs, task_assess_data]
    )

    task_generate_v2_code = Task(
        description="Generate a second version of the Python code incorporating feedback and reflections from the initial review.",
        agent=agents["Machine_Learning_Engineer"],
        expected_output="An improved version of the Python code.",
        context=[task_reflect_on_code, task_generate_code]
    )

    task_explain_code = Task(
        description="Provide a detailed explanation of the final version of the Python code, highlighting key components and their functionality.",
        agent=agents["Machine_Learning_Engineer"],
        expected_output="A comprehensive explanation of the final Python code.",
        context=[task_generate_v2_code, task_research_docs]
    )

    return [
        task_define_problem,
        task_assess_data,
        task_recommend_model,
        task_research_docs,
        task_generate_code,
        task_reflect_on_code,
        task_generate_v2_code,
        task_explain_code
    ]
