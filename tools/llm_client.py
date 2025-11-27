import os
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMClient:
    _instance = None
    @staticmethod
    def get_llm():
        if LLMClient._instance is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            LLMClient._instance = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=api_key
            )
        return LLMClient._instance
