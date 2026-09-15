"""LangGraph pipeline: search for each competitor's website, fetch its content,
extract structured info, loop until all are done, then compile one combined briefing."""

import os
import operator
import re
from typing import TypedDict, Annotated
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Just names now — the search step finds each company's URL automatically.
# Includes one made-up name to test the "search finds nothing useful" case.
COMPANIES = ["Airtable", "Superhuman", "Zzzznonexistentcompanyxyz123"]

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

llm = ChatGroq(model="openai/gpt-oss-120b")
structured_llm = llm.with_structured_output(CompetitorInfo)
search_tool = DuckDuckGoSearchResults(output_format="list", max_results=5)


def get_homepage_url(link: str) -> str:
    """Trim a search result link down to just its homepage (scheme + domain)."""
    parsed = urlparse(link)
    return f"{parsed.scheme}://{parsed.netloc}"


def pick_best_result(name: str, results: list[dict]) -> dict:
    """Prefer a result whose domain actually contains the company name
    (e.g. avoid a Wikipedia article outranking the real homepage)."""
    name_key = re.sub(r"[^a-z0-9]", "", name.lower())
    for result in results:
        domain_key = re.sub(r"[^a-z0-9]", "", urlparse(result["link"]).netloc.lower())
        if name_key and name_key in domain_key:
            return result
    return results[0]  # fallback: no domain matched, just use the top result


# --- Graph state ---
# companies_queue: company names left to process (overwritten each step)
# current: the company being worked on right now, e.g. {"name": ..., "url": ...}
# reports: accumulated results (each node adds to this list)
class GraphState(TypedDict):
    companies_queue: list[str]
    current: dict
    current_text: str
    fetch_error: str | None
    reports: Annotated[list[CompetitorInfo], operator.add]


def make_failed_report(name: str, reason: str) -> CompetitorInfo:
    note = f"Data not found ({reason})"
    return CompetitorInfo(
        competitor_name=name,
        pricing_model=note,
        core_features=[note],
        market_positioning=note,
    )


# --- Nodes ---
def search_node(state: GraphState) -> dict:
    queue = state["companies_queue"]
    name = queue[0]
    remaining = queue[1:]

    print(f"Searching: {name}'s official website...")
    try:
        results = search_tool.invoke(f"{name} official website")
        if not results:
            raise ValueError("no search results found")
        best = pick_best_result(name, results)
        url = get_homepage_url(best["link"])
        print(f"  Found: {url}")
        return {
            "companies_queue": remaining,
            "current": {"name": name, "url": url},
            "fetch_error": None,
        }
    except Exception as e:
        print(f"  Search failed for {name}: {e}")
        return {
            "companies_queue": remaining,
            "current": {"name": name, "url": None},
            "fetch_error": f"search failed: {e}",
        }


def fetch_node(state: GraphState) -> dict:
    company = state["current"]

    if state["fetch_error"]:
        # Search already failed — nothing to fetch.
        return {"current_text": ""}

    print(f"Fetching: {company['name']} ({company['url']})")
    try:
        loader = WebBaseLoader(company["url"])
        docs = loader.load()
        page_text = docs[0].page_content[:5000]
        return {"current_text": page_text, "fetch_error": None}
    except Exception as e:
        print(f"  Failed to fetch {company['name']}: {e}")
        return {"current_text": "", "fetch_error": str(e)}


def extract_node(state: GraphState) -> dict:
    company = state["current"]

    if state["fetch_error"]:
        print(f"Skipping extraction for {company['name']} (no page available)")
        return {"reports": [make_failed_report(company["name"], "search or fetch failed")]}

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
        return {"reports": [make_failed_report(company["name"], "LLM call failed")]}


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
    return "search_node" if state["companies_queue"] else "compile_node"


# --- Build graph ---
builder = StateGraph(GraphState)
builder.add_node("search_node", search_node)
builder.add_node("fetch_node", fetch_node)
builder.add_node("extract_node", extract_node)
builder.add_node("compile_node", compile_node)

builder.add_edge(START, "search_node")
builder.add_edge("search_node", "fetch_node")
builder.add_edge("fetch_node", "extract_node")
builder.add_conditional_edges("extract_node", route_after_extract)
builder.add_edge("compile_node", END)

graph = builder.compile()


if __name__ == "__main__":
    graph.invoke({"companies_queue": COMPANIES, "reports": []})
