"""Streamlit UI for the competitor briefing agent.

Same overall shape as a typical LangGraph + Streamlit market research app:
node functions report progress via st.write(), the graph runs on button click,
and results are shown as expandable cards. Unlike a version with a search API,
here the user supplies competitor names + URLs directly (no auto-discovery,
no news search) and pages are fetched straight from each competitor's site.
"""

import os
import operator
import re
from typing import Annotated, TypedDict

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
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
    companies_queue: list[dict]
    current: dict
    current_text: str
    fetch_error: str | None
    reports: Annotated[list[CompetitorInfo], operator.add]


_MARKDOWN_SPECIAL_CHARS = re.compile(r"([\\`*_{}\[\]()#+\-.!$~<>|])")


def escape_markdown(text: str) -> str:
    """Escape Markdown-special characters so LLM-generated text renders as plain text."""
    return _MARKDOWN_SPECIAL_CHARS.sub(r"\\\1", text)


def make_failed_report(company: dict, reason: str) -> CompetitorInfo:
    note = f"Data not found ({reason})"
    return CompetitorInfo(
        competitor_name=company["name"],
        pricing_model=note,
        core_features=[note],
        market_positioning=note,
    )


def build_graph(structured_llm):
    """Build the fetch -> extract -> loop -> compile graph, with nodes that
    report progress into the Streamlit UI."""

    def fetch_node(state: GraphState) -> dict:
        queue = state["companies_queue"]
        company = queue[0]
        remaining = queue[1:]

        st.write(f"🌐 **Researcher:** Fetching `{company['name']}` ({company['url']})...")
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
            st.warning(f"Could not fetch {company['name']}: {e}")
            return {
                "companies_queue": remaining,
                "current": company,
                "current_text": "",
                "fetch_error": str(e),
            }

    def extract_node(state: GraphState) -> dict:
        company = state["current"]

        if state["fetch_error"]:
            st.write(f"⏭️ Skipping extraction for `{company['name']}` (fetch failed)")
            return {"reports": [make_failed_report(company, "page fetch failed")]}

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
            return {"reports": [make_failed_report(company, "LLM call failed")]}

    def route_after_extract(state: GraphState) -> str:
        return "fetch_node" if state["companies_queue"] else "compile_node"

    def compile_node(state: GraphState) -> dict:
        return {}

    builder = StateGraph(GraphState)
    builder.add_node("fetch_node", fetch_node)
    builder.add_node("extract_node", extract_node)
    builder.add_node("compile_node", compile_node)

    builder.add_edge(START, "fetch_node")
    builder.add_edge("fetch_node", "extract_node")
    builder.add_conditional_edges("extract_node", route_after_extract)
    builder.add_edge("compile_node", END)

    return builder.compile()


def main():
    st.set_page_config(page_title="Competitor Briefing Agent", page_icon="🔍")
    st.title("🔍 Competitor Briefing Agent")
    st.markdown("Enter 2-3 competitors (name + website URL) to generate a structured briefing.")

    companies = []
    for i in range(1, 4):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(f"Competitor {i} name", key=f"name_{i}")
        with col2:
            url = st.text_input(
                f"Competitor {i} URL", key=f"url_{i}", placeholder="https://..."
            )
        if name and url:
            companies.append({"name": name, "url": url})

    if st.button("Run Research Pipeline"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found in .env file.")
            return

        if len(companies) < 2:
            st.warning("Please enter at least 2 competitors (name + URL).")
            return

        llm = ChatGroq(model="openai/gpt-oss-120b")
        structured_llm = llm.with_structured_output(CompetitorInfo)
        graph = build_graph(structured_llm)

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
