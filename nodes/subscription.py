import os
import sys
from crewai import Agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_provider import call_gemini
from utils.chat_history import add_to_session_history, get_session_history, get_all_session_history
from billing import BillingAgent

class SubscriptionAgent:
    def subscription_agent(self):
        """
        An agent that handles subscription-related queries.
        """
        return Agent(
            role="You are a helpful customer service assistant for a telecom company called Zephlen. You handle subscription-related queries.",
            goal="Your task is to assist the user with subscription issues, such as plan changes, new subscriptions, and cancellations but before that you need to authenticate the user. If the issue is not related to subscriptions, guide the user to the General Assistant.",
            keywords=["subscription", "plan", "package", "change plan", "new subscription", "cancel subscription", "upgrade", "downgrade",
                      "abonelik", "paket", "plan değişikliği", "yeni abonelik", "abonelik iptali"],
            tools={
                "call_gemini": call_gemini,
                "add_to_session_history": add_to_session_history,
                "get_session_history": get_session_history,
                "get_all_session_history": get_all_session_history,
                "get_user_tc_kimklik": BillingAgent.get_user_tc_kimklik,
                "authentication_agent": BillingAgent().authentication_agent()
            },
            max_iterations=5,
            max_execution_time=300,  # 5 minutes
            temperature=0.2
        )