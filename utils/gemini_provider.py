import os 
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
logger = logging.getLogger(__name__)

async def call_gemini(prompt:str, system_message:Optional[str]=None, temperature=0.1, max_tokens: int = 2048) -> str:
    """
    Calling Gemini.
    
    Args:
        Propmt: User Prompt for Gemini.
        system_message: Prompt to guide gemini for his rule.
        temperature: Model Creativity.
        max_tokens: Model response length.

    returns:
        str: Gemini response.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    # Initialize the model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60.0
    )

    # Create the full prompt
    full_prompt = []
    if system_message:
        full_prompt.append(system_message)
    full_prompt.append(prompt)

    # Call the model
    response = await model.ainvoke([HumanMessage(content="\n\n".join(full_prompt))])

    logger.debug(f"Gemini Call successful:  - prompt length: {len(prompt)}, response length: {len(response.content)}")
    return response.content.strip()

async def test_call_gemini() -> bool:
    """
    Test if Gemini is working as expected.

    Returns: 
        bool: True if Gemini is working, False otherwise.
    """
    response = await call_gemini("Test Connection - respond with 'ok'")

    if response == 'ok':
        return True
    else:
        logger.error(f"Gemini connection test failed: {response}")
        return False
    
if __name__ == "__main__":
    import asyncio

    # Test
    async def main():
        logging.basicConfig(level=logging.DEBUG)

        print("Test Connection for Gemini")

        if await test_call_gemini():
            print("Gemini is working correctly.")
        else:
            print("Gemini is not working correctly.")

        # Test Gemini with a sample prompt
        response = await call_gemini("Hello, Gemini!")
        print(f"Gemini response: {response}")

    asyncio.run(main())