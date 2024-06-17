from crewai_tools import tool
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import logging
from textwrap import dedent
from bs4 import BeautifulSoup
import arxiv
import fitz  # PyMuPDF
import pandas as pd
import sweetviz as sv
import streamlit as st
from typing import List, Union

api_key = os.getenv('JINA_API_KEY')

@tool("create pie plot")
def create_pie_plot(
    data: List[float], 
    labels: List[str], 
    title: str, 
    filename: str
    ) -> str:
    """
    Creates a pie plot using Matplotlib, saves it as an image, and returns its path.

    Parameters:
        data (List[float]): A list of numerical values representing the sizes of the sectors.
        labels (List[str]): A list of strings representing the labels for each sector.
        title (str): The title of the pie plot.
        filename (str): The descriptive filename for the saved image.

    Returns:
        str: The path to the saved pie plot image.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    plt.figure(figsize=(8, 8))
    patches, _, _ = plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.savefig(os.path.join("./plots", filename))
    plt.close()
    return os.path.join("./plots", filename)

@tool("create scatter plot")
def create_scatter_plot(
    x_data: Union[List[float], np.ndarray], 
    y_data: Union[List[float], np.ndarray], 
    title: str, 
    filename: str, 
    xlabel: str = None, 
    ylabel: str = None
    ) -> str:
    """
    Creates a scatter plot using Matplotlib, saves it as an image, and returns its path.

    Parameters:
        x_data (Union[List[float], np.ndarray]): The x-coordinates of the data points.
        y_data (Union[List[float], np.ndarray]): The y-coordinates of the data points.
        title (str): The title of the scatter plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        filename (str): The descriptive filename for the saved image.

    Returns:
        str: The path to the saved scatter plot image.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join("./plots", filename))
    plt.close()
    return os.path.join("./plots", filename)

@tool("create bar plot")
def create_bar_plot(
    data: Union[List[float], np.ndarray], 
    x_labels: List[str], 
    title: str, 
    filename: str, 
    xlabel: str = None, 
    ylabel: str = None
    ) -> str:
    """
    Creates a bar plot using Matplotlib, saves it as an image, and returns its path.

    Parameters:
        data (Union[List[float], np.ndarray]): The heights of the bars.
        x_labels (List[str]): The labels for the x-axis.
        title (str): The title of the bar plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        filename (str): The descriptive filename for the saved image.

    Returns:
        str: The path to the saved bar plot image.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_labels, data)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join("./plots", filename))
    plt.close()
    return os.path.join("./plots", filename)

@tool("create time series plot")
def create_time_series_plot(
    x_data: Union[List[float], np.ndarray], 
    y_data: Union[List[float], np.ndarray], 
    title: str, 
    filename: str, 
    xlabel: str = None, 
    ylabel: str = None
    ) -> str:
    """
    Creates a time series plot using Matplotlib, saves it as an image, and returns its path.

    Parameters:
        x_data (Union[List[float], np.ndarray]): The x-coordinates (time points) of the data points.
        y_data (Union[List[float], np.ndarray]): The y-coordinates (values) of the data points.
        title (str): The title of the time series plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        filename (str): The descriptive filename for the saved image.

    Returns:
        str: The path to the saved time series plot image.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join("./plots", filename))
    plt.close()
    return os.path.join("./plots", filename)

@tool("create heat map")
def create_heatmap(
    data: Union[np.ndarray, List[List[float]]], 
    x_labels: List[str], 
    y_labels: List[str], 
    title: str, 
    filename: str, 
    xlabel: str = None, 
    ylabel: str = None
    ) -> str:
    """
    Creates a heatmap using Matplotlib, saves it as an image, and returns its path.

    Parameters:
        data (Union[np.ndarray, List[List[float]]]): The data values for the heatmap.
        x_labels (List[str]): Labels for the x-axis.
        y_labels (List[str]): Labels for the y-axis.
        title (str): The title of the heatmap.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        filename (str): The descriptive filename for the saved image.

    Returns:
        str: The path to the saved heatmap image.
    """
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.savefig(os.path.join("./plots", filename))
    plt.close()
    return os.path.join("./plots", filename)

@tool("web search")
def perform_web_search(query: str) -> str:
    """
    Perform a web search using Jina AI's Reader and Searcher tools.

    This function performs a web search for the given query using the Jina AI Searcher API.
    It fetches the top search result and retrieves the content from the URL using the
    Jina AI Reader API, returning the raw content from the URL with resized images.

    Args:
        query (str): The search query.

    Returns:
        str: Content of the top search result with resized images.
    """
    reader_base_url = "https://r.jina.ai/"
    searcher_base_url = "https://s.jina.ai/"

    try:
        encoded_query = requests.utils.quote(query)
        search_url = f"{searcher_base_url}{encoded_query}"
        
        headers = {"Authorization": f"Bearer {api_key}"}
        search_response = requests.get(search_url, headers=headers)
        
        if search_response.status_code == 402:
            logging.error("Search request failed with status code 402: Payment required. Check your API key and subscription.")
            return "Error: Payment required. Check your API key and subscription."
        
        if search_response.status_code != 200:
            logging.error(f"Search request failed with status code {search_response.status_code}")
            return f"Error: Search request failed with status code {search_response.status_code}"
        
        search_results_text = search_response.text.split('\n')
        top_result_url = None

        for result in search_results_text:
            if "URL Source:" in result:
                url_start = result.find("https://")
                if url_start != -1:
                    top_result_url = result[url_start:].split()[0]
                    break

        if not top_result_url:
            logging.error("No valid URL found in the search results.")
            return "Error: No valid URL found in the search results."

        try:
            reader_url = f"{reader_base_url}{top_result_url}"
            reader_response = requests.get(reader_url, headers=headers)
            if reader_response.status_code == 200:
                content = reader_response.text
                if content:
                    # Parse HTML content and resize images
                    soup = BeautifulSoup(content, 'html.parser')
                    for img in soup.find_all('img'):
                        img['style'] = 'max-width:100%;height:auto;'
                    return str(soup)
                else:
                    logging.error(f"Empty content fetched from {top_result_url}")
                    return "Error: Empty content fetched from the URL."
            else:
                logging.error(f"Failed to fetch content from {top_result_url} with status code {reader_response.status_code}")
                return f"Error: Failed to fetch content from the URL with status code {reader_response.status_code}"
        except requests.RequestException as e:
            logging.error(f"Error fetching content from {top_result_url}: {e}")
            return f"Error fetching content from the URL: {e}"

    except requests.RequestException as e:
        logging.error(f"Error performing search: {e}")
        return f"Error performing search: {e}"

@tool("markdown cheat sheet")
def markdown_cheat_sheet() -> str:
    """
    This tool provides a markdown cheat sheet with examples of various formatting options.

    Returns:
        str: A markdown cheat sheet with examples.
    """
    return dedent("""
    ### Headers

    # Header 1
    ## Header 2
    ### Header 3

    ### Text Formatting
    
    **Bold Text**
    *Italic Text*
    ~~Strikethrough Text~~

    ### Lists

    - Item 1
    - Item 2
      - Sub-item 1
      - Sub-item 2

    ### Blockquotes
    
    > This is a blockquote.
    
    ### Code Blocks
    
    ```python
    print("Hello, World!")
    ```
    
    ### Tables
    
    | Header 1 | Header 2 |
    |----------|----------|
    | Data 1   | Data 2   |
    
    ### Horizontal Lines
    
    ---
    
    ### Inline HTML
    
    You can also use HTML directly in markdown for more complex formatting.
    
    Remember to replace placeholders with your actual content. Markdown is quite flexible, so you can mix and match these elements as needed to create your report.
    """)

def download_and_extract_pdf(
    url: str, 
    max_pages: int = 2
    ) -> str:
    """
    Downloads a PDF from the given URL and extracts text from the first few pages.

    Parameters:
        url (str): The URL of the PDF to download.
        max_pages (int): The maximum number of pages to extract text from. Default is 2.

    Returns:
        str: The extracted text from the PDF.
    """
    response = requests.get(url)
    filename = 'temp.pdf'
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    text = ""
    with fitz.open(filename) as doc:
        for page_num in range(min(max_pages, doc.page_count)):
            page = doc.load_page(page_num)
            text += page.get_text()
    
    return text

client = arxiv.Client()

@tool("Search Arxiv research papers")
def search_arxiv(query: str) -> str:
    """
    Search for research papers on arXiv and return the results in Markdown format.

    Args:
        query (str): The search query to find relevant papers.

    Returns:
        str: A string containing the search results formatted in Markdown.

    The output includes:
    - Paper title
    - Authors
    - Publication date
    - Summary
    - Extracted content from the first page of the PDF
    - Link to the full PDF

    Example usage:
    ```python
    markdown_results = search_arxiv("machine learning")
    print(markdown_results)
    ```
    """
    max_results = 10
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = client.results(search)
    
    markdown_output = ""
    
    for result in results:
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published': result.published,
            'pdf_url': result.pdf_url
        }
        
        paper_text = download_and_extract_pdf(paper_info['pdf_url'])
        
        markdown_output += f"## {paper_info['title']}\n"
        markdown_output += f"**Authors**: {', '.join(paper_info['authors'])}\n\n"
        markdown_output += f"**Published**: {paper_info['published']}\n\n"
        markdown_output += f"**Summary**: {paper_info['summary']}\n\n"
        markdown_output += f"**Content**: {paper_text[:2000]}...\n\n"
        markdown_output += f"[PDF Link]({paper_info['pdf_url']})\n\n"
        markdown_output += "---\n\n"
    
    return markdown_output
