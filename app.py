"""Streamlit UI for the competitor briefing agent.

Same overall shape as a typical LangGraph + Streamlit market research app:
node functions report progress via st.write(), the graph runs on button click,
and results are shown as expandable cards. The user only types competitor
names; a DuckDuckGo search step finds each one's website automatically.
"""

import os
import operator
import re
from typing import Annotated, TypedDict
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()


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


class GraphState(TypedDict):
    companies_queue: list[str]
    current: dict
    current_text: str
    fetch_error: str | None
    reports: Annotated[list[CompetitorInfo], operator.add]


_MARKDOWN_SPECIAL_CHARS = re.compile(r"([\\`*_{}\[\]()#+\-.!$~<>|])")


def escape_markdown(text: str) -> str:
    """Escape Markdown-special characters so LLM-generated text renders as plain text."""
    return _MARKDOWN_SPECIAL_CHARS.sub(r"\\\1", text)


def make_failed_report(name: str, reason: str) -> CompetitorInfo:
    note = f"Data not found ({reason})"
    return CompetitorInfo(
        competitor_name=name,
        pricing_model=note,
        core_features=[note],
        market_positioning=note,
    )


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


def build_graph(structured_llm, search_tool):
    """Build the search -> fetch -> extract -> loop -> compile graph, with
    nodes that report progress into the Streamlit UI."""

    def search_node(state: GraphState) -> dict:
        queue = state["companies_queue"]
        name = queue[0]
        remaining = queue[1:]

        st.write(f"🔍 **Search:** Finding `{name}`'s website...")
        try:
            results = search_tool.invoke(f"{name} official website")
            if not results:
                raise ValueError("no search results found")
            best = pick_best_result(name, results)
            url = get_homepage_url(best["link"])
            return {
                "companies_queue": remaining,
                "current": {"name": name, "url": url},
                "fetch_error": None,
            }
        except Exception as e:
            st.warning(f"Search failed for {name}: {e}")
            return {
                "companies_queue": remaining,
                "current": {"name": name, "url": None},
                "fetch_error": f"search failed: {e}",
            }

    def fetch_node(state: GraphState) -> dict:
        company = state["current"]

        if state["fetch_error"]:
            return {"current_text": ""}

        st.write(f"🌐 **Researcher:** Fetching `{company['name']}` ({company['url']})...")
        try:
            loader = WebBaseLoader(company["url"])
            docs = loader.load()
            page_text = docs[0].page_content[:5000]
            return {"current_text": page_text, "fetch_error": None}
        except Exception as e:
            st.warning(f"Could not fetch {company['name']}: {e}")
            return {"current_text": "", "fetch_error": str(e)}

    def extract_node(state: GraphState) -> dict:
        company = state["current"]

        if state["fetch_error"]:
            st.write(f"⏭️ Skipping extraction for `{company['name']}` (no page available)")
            return {"reports": [make_failed_report(company["name"], "search or fetch failed")]}

        st.write(f"📊 **Analyst:** Extracting info for `{company['name']}`...")
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
            st.warning(f"LLM extraction failed for {company['name']}: {e}")
            return {"reports": [make_failed_report(company["name"], "LLM call failed")]}

    def route_after_extract(state: GraphState) -> str:
        return "search_node" if state["companies_queue"] else "compile_node"

    def compile_node(state: GraphState) -> dict:
        return {}

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

    return builder.compile()


def main():
    st.set_page_config(page_title="Competitor Briefing Agent", page_icon="🔍")
    st.title("🔍 Competitor Briefing Agent")
    st.markdown(
        "Enter 2-3 competitor names — we'll find each one's website automatically "
        "and generate a structured briefing."
    )

    companies = []
    for i in range(1, 4):
        name = st.text_input(f"Competitor {i} name", key=f"name_{i}")
        if name:
            companies.append(name)

    if st.button("Run Research Pipeline"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found in .env file.")
            return

        if len(companies) < 2:
            st.warning("Please enter at least 2 competitor names.")
            return

        llm = ChatGroq(model="openai/gpt-oss-120b")
        structured_llm = llm.with_structured_output(CompetitorInfo)
        search_tool = DuckDuckGoSearchResults(output_format="list", max_results=5)
        graph = build_graph(structured_llm, search_tool)

        with st.status("Agent pipeline running...", expanded=True) as status:
            final_state = graph.invoke({"companies_queue": companies, "reports": []})
            status.update(label="Research complete!", state="complete", expanded=False)

        st.divider()
        st.header("🎯 Competitor Briefing")

        for report in final_state["reports"]:
            with st.expander(f"🏁 {escape_markdown(report.competitor_name)}", expanded=True):
                st.subheader("Pricing")
                st.write(escape_markdown(report.pricing_model))

                st.subheader("Core Features")
                for feature in report.core_features:
                    st.markdown(f"- {escape_markdown(feature)}")

                st.subheader("Market Positioning")
                st.write(escape_markdown(report.market_positioning))


if __name__ == "__main__":
    main()
