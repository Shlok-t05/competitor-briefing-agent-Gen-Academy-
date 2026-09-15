"""Quick sanity check: does our Gemini API key work?"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # reads GOOGLE_API_KEY from the .env file into the environment

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_tokens=200)
response = llm.invoke("Say 'API key works!' and nothing else.")

print(response.text)
