"""Fetch one webpage and print its extracted text content."""

import os

os.environ.setdefault("USER_AGENT", "competitor-briefing-agent/0.1 (course project)")

from langchain_community.document_loaders import WebBaseLoader

URL = "https://www.python.org"  # swap this for a real competitor URL later

loader = WebBaseLoader(URL)
docs = loader.load()

page_text = docs[0].page_content
print(f"Fetched {len(page_text)} characters from {URL}")
print("---- First 500 characters ----")
print(page_text[:500])
