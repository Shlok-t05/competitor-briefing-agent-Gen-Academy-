"""Quick sanity check: does our Groq API key work?"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from the .env file into the environment

llm = ChatGroq(model="openai/gpt-oss-120b")
response = llm.invoke("Say 'API key works!' and nothing else.")

print(response.text)
