# app.py
import streamlit as st
import pandas as pd
from crewai import Crew, Process
from langchain_groq import ChatGroq
from tasks import setup_tasks
from agents import initialize_agents
from streamlitHelpers import create_sidebar, create_streamlit_UI
from tools import *
import streamlit.components.v1 as components  # Importing the components module
import os
import sweetviz as sv

st.set_page_config(layout="wide")


def reports(df):
    report = sv.analyze(df)
    report.show_html("data_assessment.html", open_browser=False)


def main():

    # Initialize the language model
    llm = ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("MODEL"),
    )
    create_streamlit_UI(
        "Your Machine Learning Assistant",
        "Describe your machine learning problem and upload a CSV file with your data.",
    )

    user_question = st.text_input(
        "Describe your ML problem:"
    )  # Input for ML problem description
    uploaded_file = st.file_uploader(
        "Upload a sample .csv of your data (optional)", type=["csv"]
    )  # File uploader for CSV file

    if user_question and uploaded_file:

        df = pd.read_csv(uploaded_file)  # Read the CSV file into a DataFrame
        agents = initialize_agents(llm)  # Initialize agents with the language model
        reports(df)  # Display the HTML report
        with st.chat_message("Data_Assessment_Agent", avatar="📊"):
            with open("data_assessment.html", "r") as f:
                components.html(f.read(), height=800, scrolling=True)
        tasks = setup_tasks(
            agents, user_question, df, uploaded_file
        )  # Setup tasks with the defined agents and inputs

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,  # Define the process as sequential
            full_output=True,  # Return the full output with all tasks' outputs
            verbose=True,
        )

        result = crew.kickoff()  # Execute the tasks

        # st.write(result)  # Output the result in Streamlit


if __name__ == "__main__":
    main()
