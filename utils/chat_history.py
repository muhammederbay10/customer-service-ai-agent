from datetime import datetime
import json
import re 
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
    conversation = get_all_session_history({"chat_history": recent_messages}, number=False)
    system_message = """
    You are an expert assistant that summarizes conversations between a user and an AI assistant. Summarize the conversation between the telecom company and the user.
    
    IMPORTANT INFORMATION:
    - Keep the summary concise and to the point.
    - IS ID verified or not.
    - Which plan the user is on.
    - Any specific issues or requests made by the user.
    - Any resolutions or actions taken by the assistant.
    - Do not include any irrelevant details.
    - Is the user happy or not.

    Example Summary:
    "Id is verified. Mr.Smith informed that he has a billing issue. and the user is not happy." 
    "New user registration, id not verified yet." 
    "Technical support → Appointment requested → Waiting for slot selection."

    rules:
    - Summarize in 1-2 sentences between 100-150 characters.
    """.strip()

    # Create the prompt for Gemini
    prompt = f"Summarize the following conversation:\n\n{conversation}\n\nSummary:".strip()

    # Call Gemini to get the summary
    summary = await call_gemini(prompt=prompt, system_message=system_message, temperature=0.3, max_tokens=300)
    
    # Return the summary or a default message if summarization fails
    if summary:
        return summary
    else:
        return "Failed to generate summary."
    
def get_context_for_gemini(state:Dict, history: bool=True) -> str:
    """
    Get the context for Gemini, including the current state and optionally the session history.
    """
    if not history:
        return f"Current State: {state.get('current_state', 'default')}"
    
    session_history = get_all_session_history(state, 3)

    if session_history == "No history available.":
        return f"Current State: {state.get('current_state', 'default')}"
    
    return f"""
        Current State: {state.get('current_state', 'default')}"""

async def get_last_user_message(state:Dict, role:str, message:str, summary_only:str="chat_summary", batch_size:int=15, tail_size:int=6) -> str:
    """
    Add a message to the session history.
    Args:
        state (Dict): The current state of the session.
        role (str): The role of the message sender ('user' or 'assistant').
        message (str): The content of the message.
        summary_only (str, optional): The key for the summary in the state. Defaults to "chat_summary".
        batch_size (int, optional): The number of messages to consider for summarization. Defaults to 15.
        tail_size (int, optional): The number of recent messages to keep in detail. Defaults to 6.

    Returns:
        str: The message and the response.
    """
    history = state.get("history", [])

    # Add the new message to history
    message_entry = {
        "role": role,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state.get("current_state", "default")
    }
    history.append(message_entry)
    state["chat_history"] = history

    summary = state.get(summary_only, "")

    new_messages = f"{role}: {message}\n"
    updated_summary = summary + (f"{summary}\n" if summary else "") + new_messages

    if len(history) > batch_size:
        batch_summary = await summarize_session_history([{"role": "", "messages": updated_summary}])

        # tail messages to keep in detail
        tail_messages = history[-tail_size:]
        tail_chat = "\n".join(
            [f"{entry['role']}: {entry['message']}" for entry in tail_messages]
        )

        new_summary = batch_summary.strip() + "\n" + tail_chat.strip()
        state[summary_only] = new_summary.strip()
    
    else:
        state[summary_only] = updated_summary.strip()

async def summarize_chat_history(messages:List[Dict[str, str]]) -> str:
    """
    Summarize the chat history for the given messages.
    
    Args:
        messages (List[Dict[str, str]]): The list of messages to summarize.
    
    Returns:
        str: The summary of the chat history.
    """
    if not messages: 
        return "No messages to summarize."
    
    conversation = "\n".join(
        [f"{msg['role']}: {msg['message']}" for msg in messages]
    )

    system_message = """
    You are an expert assistant that summarizes conversations between a user and an AI assistant. 
    Summarize the conversation between the telecom company and the user. And List the key points discussed in the conversation.
    {conversation}
    
    format:
    {{
        "summary": "short summary of the conversation",
    }}
    """

    prompt = f"Summarize the following conversation:\n\n{conversation}\n\nSummary:".strip()

    # Call Gemini to get the summary
    summary_prompt = await call_gemini(prompt=prompt, system_message=system_message, temperature=0.3)

    response = await call_gemini(summary_prompt)

    data = extract_json_from_response(response)
    summary = data.get("summary", "Failed to generate summary.")
    return summary

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON object from a string response.

    Args:
        response (str): The string response containing JSON.

    Returns:
        Dict[str, Any]: The extracted JSON object or an empty dictionary if extraction fails.
    """
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"tool_groups"[^{}]*\})',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        return {}