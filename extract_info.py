"""Fetch one competitor's page, then use the LLM to extract structured info from it."""

import os

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq

load_dotenv()

COMPANY_NAME = "Python.org"  # placeholder, swap for a real competitor later
URL = "https://www.python.org"


class CompetitorInfo(BaseModel):
    """Structured facts extracted about one competitor."""

    competitor_name: str = Field(description="Name of the competitor")
    pricing_model: str = Field(
        description="How they monetize, specific prices if found. 'Data not found' if unclear."
    )
    core_features: list[str] = Field(
        description="3-5 main features or offerings. ['Data not found'] if unclear."
    )
    market_positioning: str = Field(
        description="Market segment, target users, and stated advantages. 'Data not found' if unclear."
    )


SYSTEM_PROMPT = """You are an elite market analyst. Extract the requested information \
from the raw web page text below. If you cannot find a specific detail, write \
'Data not found' for that field. Do not invent facts. Do not exaggerate. \
Report only what is present in the text."""

# Step 1: fetch the page
loader = WebBaseLoader(URL)
docs = loader.load()
page_text = docs[0].page_content[:5000]  # keep it short for now

# Step 2: extract structured info with the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")
structured_llm = llm.with_structured_output(CompetitorInfo)

result = structured_llm.invoke(
    [
        ("system", SYSTEM_PROMPT),
        ("human", f"Competitor name: {COMPANY_NAME}\n\nWeb page text:\n{page_text}"),
    ]
)

print(result)
