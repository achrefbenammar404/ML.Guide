import streamlit as st 
import streamlit as st
import re
import PIL 
import logging 
import os 
import json 
from typing import Union, List, Tuple, Dict , Any , Callable
import json


def create_sidebar(title = "Select LLM|Input Groq API Key") : 
    st.sidebar.title(title)
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768',  'llama3-70b-8192', 'llama3-8b-8192', 'gemma-7b-it']
    )

    groq_api_key = st.sidebar.text_input("Groq-API-Key")
    jina_reader_api_key = st.sidebar.text_input("Jina Reader API Key")
    return model , groq_api_key ,jina_reader_api_key


def create_streamlit_UI (title , explanation_text ) : 
    st.title(title)
    st.markdown(explanation_text , unsafe_allow_html=True)


def is_valid_json(json_string : str ) -> Dict:
    try:
        json.loads(json_string)
    except ValueError:
        return False
    return True

def extract_info_from_action(input_str):
    """extracts the thought, tool input , tool output and tool from streamlit callback"""
    # Extract tool
    tool_match = re.search(r"tool='(.*?)'", input_str)
    tool = tool_match.group(1) if tool_match else None

    # Extract the entire tool input as a JSON or string, handling different newline characters
    tool_input_match = re.search(r"tool_input='(.*?)(?:'[\s,])", input_str, re.DOTALL)
    if tool_input_match:
        tool_input = tool_input_match.group(1).strip()
    else:
        tool_input = None
    # Extract thought, more lenient about trailing spaces and newline characters
    thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", input_str)
    thought = thought_match.group(1).strip() if thought_match else None

    return tool, tool_input, thought


agent_finishes  = []
def create_streamlit_callback( agent_role : str , agent_avatar  : str ) -> Callable:
    """creates a custom callback for each agent with its appropriate avatar and name"""
    def streamlit_callback(step_output ) -> None :
        if isinstance(step_output, list):
            for step in step_output:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    tool, tool_input, thought = extract_info_from_action(str(action))

                    # Initialize an empty list to collect display messages
                    display_messages = []

                    if thought:
                        thought_display = f"**{agent_role.title()}:** {thought}"
                        display_messages.append(thought_display)

                    if tool:
                        tool_display = f"**Tool Used:** {tool}"
                        display_messages.append(tool_display)

                    if tool_input:
                        tool_input_display = f"**Tool Input:** {tool_input}"
                        display_messages.append(tool_input_display)

                    # Display messages in Streamlit chat if there's anything to display
                    if display_messages:

                        with st.chat_message(agent_role, avatar=agent_avatar):
                            for message in display_messages:
                                st.markdown(message)
                            with st.expander("See Tool Result : ") :
                                if ( is_image_path(observation)) :
                                    st.write(PIL.Image.open(observation))
                                else :
                                    if is_valid_json(observation):
                                        st.json(observation)
                                    else:
                                        st.write(observation)
                else:
                    # For non-tuple or unexpected data, handle it here
                    with st.chat_message(agent_role, avatar=agent_avatar):
                        st.markdown(step)
        else:
            #print(type(step_output))
            with st.chat_message(agent_role, avatar=agent_avatar):
                st.markdown(f"**{agent_role.title()}**")
                st.markdown("**finshed task** :  ")
                st.markdown(f"{step_output.return_values['output']}")

    return streamlit_callback


def is_image_path(path : str ) -> bool:
    """
    Checks if the provided string path points to a valid image file based on its extension.

    Parameters:
        - path (str): The file path or filename to check.

    Returns:
        - bool: True if the path is an image file, False otherwise.

    Example Usage:
        result = is_image_path("example.jpg")  # Returns: True
        result = is_image_path("document.txt") # Returns: False
        result = is_image_path("photo.png")    # Returns: True
    """
    # List of valid image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Extract the extension from the path
    _, ext = os.path.splitext(path)
    
    # Check if the extension is in the list of image file extensions
    return ext.lower() in image_extensions



    

            
            

