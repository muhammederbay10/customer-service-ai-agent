from datetime import datetime
import json 
import os
import sys
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_provider import call_gemini

def add_to_session_history(state:Dict, role:str, message:str, current_state:str=None) -> List[Dict[str, Any]]:
    """
    Add a message to the session history.

    Args:
        state (Dict): The current state of the session.
        role (str): The role of the message sender ('user' or 'assistant').
        message (str): The content of the message.
        current_state (str, optional): The current state description. Defaults to None.
    """
    history = state.get("history", [])

    message_entry = {
        "role": role,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": current_state if current_state else state.get("current_state", "default")
    }

    history.append(message_entry)
    return history

def get_session_history(state:Dict, last_n_message:int=5) -> str:
    """
    Get the last n messages from the session history.

    Args:
        state (Dict): The current state of the session.
        last_n_message (int, optional): The number of messages to retrieve. Defaults to 5.

    Returns:
        str: The formatted session history.
    """
    history = state.get("history", [])
    relevant_history = history[-last_n_message:] if last_n_message > 0 else history

    if not relevant_history:
        return "No history available."

    formatted_history = []
    for entry in relevant_history:
        formatted_entry = "user: " + entry["message"] if entry["role"] == "user" else "assistant: " + entry["message"]
        formatted_history.append(formatted_entry)

    return "\n".join(formatted_history)

def get_all_session_history(state:Dict, number:bool=True) -> str:
    """
    Get the entire session history.

    Args:
        state (Dict): The current state of the session.
        number (bool, optional): Whether to number the messages. Defaults to True.

    Returns:
        str: The message and the response.
    """
    history = state.get("history", [])
    if not history:
        return "No history available."
    
    messages = []
    for idx, entry in enumerate(history, start=1):
        role = "user" if entry["role"] == "user" else "assistant"
        message = entry["message"]

        if number:
            messages.append(f"{idx}. {role}: {message}")
        else:
            messages.append(f"{role}: {message}")
    return "\n".join(messages)

async def summarize_session_history(state:Dict, last_n_message:int=5) -> str:
    """
    Summarize the last n messages from the session history using Gemini.
    
    Args:
        state (Dict): The current state of the session.
        last_n_message (int, optional): The number of messages to summarize. Defaults to 5.
        
    Returns:
        str: The summary of the session history.
    """
    history = state.get("history", [])
    relevant_history = history[-last_n_message:] if last_n_message > 0 else history

    if not relevant_history:
        return "No history available to summarize."
    
    # Using Gemini to summarize the history
    recent_messages = history[-last_n_message:] if last_n_message > 0 else history

    # Format messages for Gemini
    conversation = get_all_session_history(state, number=False)
    system_message = """
    You are an expert assistant that summarizes conversations between a user and an AI assistant. Summarize the conversation between the teleco company and the user."""