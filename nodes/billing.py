import os
import sys
from crewai import Agent
from typing import Dict, Optional
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_provider import call_gemini
from utils.chat_history import add_to_session_history, get_session_history, get_all_session_history
from database import db

class BillingAgent:
    def get_user_tc_kimklik(message: str) -> Optional[Dict[str, str]]:
        """
        Extract TC KİMLİK Number from the user's message.

        Args:
            message (str): The user's message.

        Returns:
            str: Extracted TC KİMLİK Number or an empty string if not found.
        """
        match = re.search(r'\b\d{11}\b', message)
        if match:
            return {"TC KİMLİK": match.group(0)}
        return None

    def authentication_agent(self):
        """
        An agent that handles the authentication process before accessing billing-related queries.
        """
        return Agent(
            role="You are a helpful customer service assistant for a telecom company called Zephlen. You handle the authentication process before accessing billing-related queries.",
            goal='Your task is to authenticate the user by asking for their TC KİMLİK Number. Once authenticated, After the authentication process, '
                 'Extract the user\'s TC Kimlik number from their message. Respond ONLY with a JSON object in the form: {"tc_kimlik": "12345678901"}. '
                 'If not found, respond with {"tc_kimlik": null}. and stop.',
            keywords=["authenticate", "authentication", "account number", "security question", "login", "verify", "identity", 
                      "doğrulama", "hesap numarası", "güvenlik sorusu", "giriş yap", "kimlik doğrulama"],
            tools={
                "call_gemini": call_gemini,
                "add_to_session_history": add_to_session_history,
                "get_session_history": get_session_history,
                "get_all_session_history": get_all_session_history,
                "get_user_tc_kimklik": BillingAgent.get_user_tc_kimklik
            },
            max_iterations=5,
            max_execution_time=300,  # 5 minutes
            temperature=0.2
        )


    def billing_agent(self):
        """
        An agent that handles billing-related queries after the authentication process.
        """
        return Agent(
            role="You are a helpful customer service assistant for a telecom company called Zephlen. You handle billing-related queries after the authentication process.",
            goal="Your task is to assist the user with billing issues, such as explaining charges, payment methods, and billing cycles. If the issue is not related to billing, guide the user to the General Assistant.",
            keywords=["billing", "payment", "invoice", "charge", "bill", "due date", "late fee", "refund", "transaction", "account balance", 
                      "fatura", "ödeme", "fatura", "ücret", "fatura tarihi", "gecikme ücreti", "geri ödeme", "işlem", "hesap bakiyesi"],
            tools={
                "call_gemini": call_gemini,
                "add_to_session_history": add_to_session_history,
                "get_session_history": get_session_history,
                "get_all_session_history": get_all_session_history
            },
            max_iterations=5,
            max_execution_time=300,  # 5 minutes
            temperature=0.2
        )
