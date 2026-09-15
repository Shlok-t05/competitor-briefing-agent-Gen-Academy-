"""LangGraph pipeline: fetch each competitor's page, extract structured info,
loop until all are done, then compile one combined briefing."""

import os
import operator
from typing import TypedDict, Annotated

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Placeholder competitors for now — swap for real ones later.
# Includes one deliberately broken URL to test failure handling.
COMPANIES = [
    {"name": "Python", "url": "https://www.python.org"},
    {"name": "Django", "url": "https://www.djangoproject.com"},
    {"name": "Broken Example", "url": "https://this-domain-does-not-exist-12345.com"},
    {"name": "Flask", "url": "https://flask.palletsprojects.com/"},
]

BRIEFING_FILE = "briefing.md"


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

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
structured_llm = llm.with_structured_output(CompetitorInfo)


# --- Graph state ---
# companies_queue: companies left to process (overwritten each step)
# current: the company being worked on right now
# reports: accumulated results (each node adds to this list)
class GraphState(TypedDict):
    companies_queue: list[dict]
    current: dict
    current_text: str
    fetch_error: str | None
    reports: Annotated[list[CompetitorInfo], operator.add]


# --- Nodes ---
def fetch_node(state: GraphState) -> dict:
    queue = state["companies_queue"]
    company = queue[0]
    remaining = queue[1:]

    print(f"Fetching: {company['name']} ({company['url']})")
    try:
        loader = WebBaseLoader(company["url"])
        docs = loader.load()
        page_text = docs[0].page_content[:5000]
        return {
            "companies_queue": remaining,
            "current": company,
            "current_text": page_text,
            "fetch_error": None,
        }
    except Exception as e:
        print(f"  Failed to fetch {company['name']}: {e}")
        return {
            "companies_queue": remaining,
            "current": company,
            "current_text": "",
            "fetch_error": str(e),
        }


def make_failed_report(company: dict, reason: str) -> CompetitorInfo:
    note = f"Data not found ({reason})"
    return CompetitorInfo(
        competitor_name=company["name"],
        pricing_model=note,
        core_features=[note],
        market_positioning=note,
    )


def extract_node(state: GraphState) -> dict:
    company = state["current"]

    if state["fetch_error"]:
        print(f"Skipping extraction for {company['name']} (fetch failed)")
        return {"reports": [make_failed_report(company, "page fetch failed")]}

    print(f"Extracting: {company['name']}")
    try:
        result = structured_llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    f"Competitor name: {company['name']}\n\nWeb page text:\n{state['current_text']}",
                ),
            ]
        )
        return {"reports": [result]}
    except Exception as e:
        print(f"  LLM extraction failed for {company['name']}: {e}")
        return {"reports": [make_failed_report(company, "LLM call failed")]}


def compile_node(state: GraphState) -> dict:
    lines = ["# Competitor Briefing\n"]
    for report in state["reports"]:
        lines.append(f"## {report.competitor_name}")
        lines.append(f"**Pricing:** {report.pricing_model}")
        lines.append(f"**Core features:** {', '.join(report.core_features)}")
        lines.append(f"**Positioning:** {report.market_positioning}\n")
    briefing = "\n".join(lines)

    with open(BRIEFING_FILE, "w") as f:
        f.write(briefing)

    print(f"\nBriefing written to {BRIEFING_FILE}")
    print("\n---- Compiled briefing ----\n")
    print(briefing)
    return {}


def route_after_extract(state: GraphState) -> str:
    return "fetch_node" if state["companies_queue"] else "compile_node"


# --- Build graph ---
builder = StateGraph(GraphState)
builder.add_node("fetch_node", fetch_node)
builder.add_node("extract_node", extract_node)
builder.add_node("compile_node", compile_node)

builder.add_edge(START, "fetch_node")
builder.add_edge("fetch_node", "extract_node")
builder.add_conditional_edges("extract_node", route_after_extract)
builder.add_edge("compile_node", END)

graph = builder.compile()


if __name__ == "__main__":
    graph.invoke({"companies_queue": COMPANIES, "reports": []})
