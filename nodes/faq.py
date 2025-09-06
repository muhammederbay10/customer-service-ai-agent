import os
import sys
from typing import List
from chromadb import logger
from crewai import Agent
from qdrant_client import QdrantClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_system import embedding_system
from utils.gemini_provider import call_gemini
from billing import BillingAgent
from subscription import SubscriptionAgent

async def RAG_Chain(inputs: List[str]) -> List[str]:
    """
    Retrieve and Generate (RAG) chain to enhance responses with context.
    
    Args:
        inputs (List[str]): List of user inputs.

    Returns:
        List[str]: List of enhanced responses.    
    """
    # create embedding for user input
    input_embeddings = [embedding_system.embed_query(text) for text in inputs]

    # connect to Qdrant
    qdrant_client = QdrantClient(host="localhost", port=6333)

    search_results = qdrant_client.search(
            collection_name="turkcell_sss",
            query_vector=input_embeddings.tolist(),
            limit=3,
            with_payload=True
        )

    results = []
    for result in search_results:
        results.append({
            'score': float(result.score),
            'question': result.payload.get('question', ''),
            'answer': result.payload.get('answer', ''),
            'source': result.payload.get('source', ''),
            'relevance': 'high' if result.score > 0.8 else 'medium' if result.score > 0.6 else 'low'
                    })

    logger.info(f"Found {len(results)} relevant FAQs for question: '{inputs[0][:50]}...'")
    return results

class GeneralAgent:
    def assistant_agent(self):
        """
        An assistant agent that handles the user queries and provides relevant answers and takes them to the other agents.
        """
        return Agent(
            role="You are a helpful customer service assistant for a telecom company called Zephlen.",
            goal="Your task is to assist the user whether it is a billing issue or a subscription issue and take them to the relevant agent.",
            keywords=["billing", "subscription", "internet", "data", "call", "sms", "plan", "package", "offer", "customer service", "fatura", 
                      "abonelik", "internet", "veri", "arama", "sms", "plan", "paket", "teklif", "müşteri hizmetleri"],
            tools={
                "call_gemini": call_gemini,
                "RAG_Chain": RAG_Chain,
                "BillingAgent": BillingAgent().billing_agent(),
                "SubscriptionAgent": SubscriptionAgent().subscription_agent(),
            },
            max_iterations=10,
            max_execution_time=600,  # 10 minutes
            temperature=0.7
        ) 
