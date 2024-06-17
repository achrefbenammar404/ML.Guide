
# ML.Guide

ML Guide is a multi-LLM agent system designed to assist users with their machine learning problems. This project was developed during a 17-hour hackathon at the AI National Summit (AINS) event on June 9, 2024.

# My Project

## Here is a demo video of my project
(Drive Link to video):

[![Watch the video](app/resources/MLGuide.webp)](https://drive.google.com/uc?export=download&id=1OcY07YTmq3gJRJAm04t2u0GmtLQU9d2c)


## Features

- **Exploratory Data Analysis (EDA)**: Automatically performs an initial EDA using Sweetviz to give users insights into their dataset.
- **Research Paper Extraction**: Extracts relevant research papers from arXiv to provide users with the latest developments in the field.
- **Web Scraping for Similar Problems**: Uses JiraAI Reader and Serper API to search the web for similar machine learning problems and solutions.
- **Model Code Generation**: Generates code for the best machine learning model tailored to the user's problem using multi-LLM agents managed by CrewAI.

## Tools Used

<table>
  <tr>
    <td><img src="app/resources/crew_only_logo.png" alt="CrewAI Logo" width="100"/></td>
    <td><img src="app/resources/ArXiv_logo_2022.png" alt="arXiv Logo" width="100"/></td>
    <td><img src="app/resources/jinaai.png" alt="JinaAI Logo" width="100"/></td>
    <td><img src="app/resources/sweetviz.png" alt="Sweetviz Logo" width="100"/></td>
  </tr>
</table>

### Tool Explanations

- **CrewAI**: This platform is used for orchestrating multiple AI agents to work together efficiently, providing the backbone for managing and integrating different AI models to solve complex problems  .
- **arXiv**: A repository of electronic preprints (known as e-prints) approved for publication after moderation, but not peer-reviewed. It provides access to the latest research papers in the field of machine learning and other scientific areas .
- **JinaAI**: This tool is used for neural search, enabling efficient and effective search capabilities across different data types, which is essential for finding relevant information and solutions related to the user's problem .
- **Sweetviz**: An open-source Python library that generates beautiful, high-density visualizations of a pandas DataFrame, aiding in performing exploratory data analysis (EDA) to give users insights into their dataset .

## Installation

To get started with ML Guide, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/achrefbenammar404/AINS-ML.Guide.git
   cd AINS-ML.Guide
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add the following variables:
   ```ini
   GROQ_API_KEY = "your_Groq_api_key_here"
   JINA_API_KEY = "your_jina_api_key_here"
   ```

## Usage

To run the ML Guide application, execute the following command:
```bash
streamlit run src/app.py
```

This will start the Streamlit web application. Open the provided URL in your web browser to interact with the ML Guide interface.

### Steps to Use ML Guide:

1. **Input Problem and Dataset**: Upload your dataset and describe your machine learning problem.
2. **Exploratory Data Analysis**: View the automatic EDA report generated by Sweetviz.
3. **Research Extraction**: Review the relevant research papers extracted from arXiv.
4. **Web Search**: Examine similar problems and solutions found on the web.
5. **Model Generation**: Get the code for the best model to solve your problem, generated by the multi-LLM agents.