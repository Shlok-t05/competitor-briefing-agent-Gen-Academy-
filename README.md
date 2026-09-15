# Competitor Briefing Agent

A small LangChain + LangGraph project that takes 2-3 competitor names, finds
each one's website automatically, fetches its content, extracts
pricing/features/positioning with an LLM, and compiles everything into one
structured briefing.

## Scope

**In scope:**
- Accept competitor names (via `graph.py` for a fixed list, or `app.py` for a Streamlit form)
- Search (DuckDuckGo, no API key needed) to find each competitor's homepage automatically
- Fetch each competitor's webpage content
- Use an LLM (Groq) with structured output to extract pricing, core features, and market positioning
- Loop through all competitors, one at a time, using a LangGraph queue/router pattern
- Compile results into one combined briefing (`briefing.md` for the terminal version, expandable cards for the Streamlit version)
- Handle failures gracefully: a failed search, bad URL, or failed LLM call produces a "Data not found" placeholder for that competitor instead of crashing the whole run

**Out of scope:**
- Automatic competitor *discovery* — you still provide the names yourself; only the URL lookup is automated
- News search / recent announcements
- Saving to a database, exporting to PDF/CSV, or sending anything anywhere
- Any action taken based on the findings — the workflow stops after producing the briefing; a human reads and decides what to do with it

**Known limitation:** the search step picks the top result whose domain matches
the company name. This works well for specific product/company names (e.g.
"Airtable", "Superhuman") but can pick the wrong site for a name that's also a
common English word or overlaps with an unrelated well-known topic.

## Files

| File | Purpose |
|---|---|
| `test_api_key.py` | Sanity check that your Groq API key works |
| `fetch_page.py` | Standalone test of fetching one webpage's text |
| `extract_info.py` | Standalone test of structured LLM extraction on one page |
| `graph.py` | Full pipeline as a terminal script — loops over a hardcoded company list, writes `briefing.md` |
| `app.py` | Same pipeline behind a Streamlit UI — enter competitors in the browser, see live progress and results as cards |
| `requirements.txt` | Python package list |
| `.env.example` | Template for your API key file |

## Setup and Runbook

1. **Create and activate a virtual environment** (already done if you followed along)
   ```
   python3 -m venv venv
   ```
2. **Install dependencies**
   ```
   ./venv/bin/pip install -r requirements.txt
   ```
3. **Set up your API key**
   ```
   cp .env.example .env
   ```
   Then edit `.env` and paste your real key:
   ```
   GROQ_API_KEY=your-real-key-here
   ```
   Get a free key at console.groq.com/keys.

4. **Run it**
   - Terminal version (edit the `COMPANIES` list near the top of `graph.py` with your real competitor names first):
     ```
     ./venv/bin/python graph.py
     ```
   - Streamlit UI version:
     ```
     ./venv/bin/streamlit run app.py
     ```
     Opens in your browser. Enter 2-3 competitor names and click "Run Research Pipeline" — no URLs needed.

## Validation Checklist

**Before running:**
- [ ] `.env` exists with a real `GROQ_API_KEY`
- [ ] Virtual environment activated / using `./venv/bin/python`
- [ ] Dependencies installed (`./venv/bin/pip install -r requirements.txt`)

**After running:**
- [ ] No unhandled crash — a bad URL or failed LLM call should show "Data not found (...)" for that competitor, not stop the whole run
- [ ] `briefing.md` created (terminal version) or cards displayed (Streamlit version)
- [ ] Extracted fields look grounded in the actual page content, not invented

## Known Failure Modes

| Failure | Likely Cause | Fix |
|---|---|---|
| `GROQ_API_KEY not found` | Missing/empty `.env` | Add real key to `.env` |
| Rate limit / quota error | Groq's free tier has request limits too, though more generous than Gemini's | Wait a bit and retry; check console.groq.com for your current limits |
| A competitor shows all "Data not found" fields | Either search picked the wrong site, the page failed to fetch, or its content is JS-rendered (marketing sites often are) and `WebBaseLoader` only sees static HTML | Check what URL was found in the printed/displayed logs; try a more specific company name if search picked the wrong site |
| Search finds the wrong website | The company name is a common word or overlaps with an unrelated topic (e.g. "Flask") | Use a more specific/full company name |

## Extension Ideas

- Add retry-with-backoff before falling back on rate-limit errors
- Add a full competitor-*discovery* step (auto-suggest competitors from your own company name, not just look up URLs for names you give it)
- Export the briefing to PDF
- Cache fetched pages and search results to avoid re-fetching on repeated runs
