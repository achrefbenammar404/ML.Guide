import streamlit as st
import re
import PIL
import logging
import os
import json
from typing import Union, List, Tuple, Dict, Any, Callable


def create_sidebar(title: str = "Select LLM|Input Groq API Key") -> Tuple[str, str, str]:
    """
    Creates a sidebar with a title, model selection, and API key input fields.

    Args:
        title (str): The title of the sidebar. Default is "Select LLM|Input Groq API Key".

    Returns:
        Tuple[str, str, str]: Selected model, Groq API key, and Jina Reader API key.
    """
    st.sidebar.title(title)
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama3-70b-8192', 'llama3-8b-8192', 'gemma-7b-it']
    )

    groq_api_key = st.sidebar.text_input("Groq-API-Key")
    jina_reader_api_key = st.sidebar.text_input("Jina Reader API Key")
    return model, groq_api_key, jina_reader_api_key


def create_streamlit_UI(title: str, explanation_text: str) -> None:
    """
    Creates a Streamlit UI with a title and explanation text.

    Args:
        title (str): The title of the UI.
        explanation_text (str): The explanation text to be displayed.
    """
    st.title(title)
    st.markdown(explanation_text, unsafe_allow_html=True)


def is_valid_json(json_string: str) -> bool:
    """
    Checks if a given string is a valid JSON.

    Args:
        json_string (str): The string to be checked.

    Returns:
        bool: True if the string is a valid JSON, False otherwise.
    """
    try:
        json.loads(json_string)
    except ValueError:
        return False
    return True


def extract_info_from_action(input_str: str) -> Tuple[Union[str, None], Union[str, None], Union[str, None]]:
    """
    Extracts the thought, tool input, tool output, and tool from a Streamlit callback input string.

    Args:
        input_str (str): The input string from the Streamlit callback.

    Returns:
        Tuple[Union[str, None], Union[str, None], Union[str, None]]: Extracted tool, tool input, and thought.
    """
    tool_match = re.search(r"tool='(.*?)'", input_str)
    tool = tool_match.group(1) if tool_match else None

    tool_input_match = re.search(r"tool_input='(.*?)(?:'[\s,])", input_str, re.DOTALL)
    tool_input = tool_input_match.group(1).strip() if tool_input_match else None

    thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", input_str)
    thought = thought_match.group(1).strip() if thought_match else None

    return tool, tool_input, thought


agent_finishes = []

def create_streamlit_callback(agent_role: str, agent_avatar: str) -> Callable:
    """
    Creates a custom callback for each agent with its appropriate avatar and name.

    Args:
        agent_role (str): The role of the agent.
        agent_avatar (str): The avatar of the agent.

    Returns:
        Callable: The callback function.
    """
    def streamlit_callback(step_output: Any) -> None:
        if isinstance(step_output, list):
            for step in step_output:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    tool, tool_input, thought = extract_info_from_action(str(action))

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

                    if display_messages:
                        with st.chat_message(agent_role, avatar=agent_avatar):
                            for message in display_messages:
                                st.markdown(message)
                            with st.expander("See Tool Result:"):
                                if is_image_path(observation):
                                    st.image(PIL.Image.open(observation))
                                else:
                                    if is_valid_json(observation):
                                        st.json(observation)
                                    else:
                                        st.write(observation)
                else:
                    with st.chat_message(agent_role, avatar=agent_avatar):
                        st.markdown(step)
        else:
            with st.chat_message(agent_role, avatar=agent_avatar):
                st.markdown(f"**{agent_role.title()}**")
                st.markdown("**Finished task**:")
                st.markdown(f"{step_output.return_values['output']}")

    return streamlit_callback


def is_image_path(path: str) -> bool:
    """
    Checks if the provided string path points to a valid image file based on its extension.

    Args:
        path (str): The file path or filename to check.

    Returns:
        bool: True if the path is an image file, False otherwise.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(path)
    return ext.lower() in image_extensions
