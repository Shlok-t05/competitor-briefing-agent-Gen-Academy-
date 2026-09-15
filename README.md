# Competitor Briefing Agent

A small LangChain + LangGraph project that takes 2-3 competitor names and
website URLs, fetches each site's content, extracts pricing/features/positioning
with an LLM, and compiles everything into one structured briefing.

## Scope

**In scope:**
- Accept competitor name + URL pairs (via `graph.py` for a fixed list, or `app.py` for a Streamlit form)
- Fetch each competitor's webpage content
- Use an LLM (Gemini) with structured output to extract pricing, core features, and market positioning
- Loop through all competitors, one at a time, using a LangGraph queue/router pattern
- Compile results into one combined briefing (`briefing.md` for the terminal version, expandable cards for the Streamlit version)
- Handle failures gracefully: a bad URL or a failed LLM call produces a "Data not found" placeholder for that competitor instead of crashing the whole run

**Out of scope:**
- Automatic competitor discovery (you provide the names/URLs yourself — no search API)
- News search / recent announcements
- Saving to a database, exporting to PDF/CSV, or sending anything anywhere
- Any action taken based on the findings — the workflow stops after producing the briefing; a human reads and decides what to do with it

## Files

| File | Purpose |
|---|---|
| `test_api_key.py` | Sanity check that your Gemini API key works |
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
   GOOGLE_API_KEY=your-real-key-here
   ```
   Get a free key at aistudio.google.com/apikey.

4. **Run it**
   - Terminal version (edit the `COMPANIES` list near the top of `graph.py` with your real competitors first):
     ```
     ./venv/bin/python graph.py
     ```
   - Streamlit UI version:
     ```
     ./venv/bin/streamlit run app.py
     ```
     Opens in your browser. Enter 2-3 competitor names + URLs and click "Run Research Pipeline."

## Validation Checklist

**Before running:**
- [ ] `.env` exists with a real `GOOGLE_API_KEY`
- [ ] Virtual environment activated / using `./venv/bin/python`
- [ ] Dependencies installed (`./venv/bin/pip install -r requirements.txt`)

**After running:**
- [ ] No unhandled crash — a bad URL or failed LLM call should show "Data not found (...)" for that competitor, not stop the whole run
- [ ] `briefing.md` created (terminal version) or cards displayed (Streamlit version)
- [ ] Extracted fields look grounded in the actual page content, not invented

## Known Failure Modes

| Failure | Likely Cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY not found` | Missing/empty `.env` | Add real key to `.env` |
| `429 RESOURCE_EXHAUSTED` | Gemini free tier is capped at 20 requests/day per model | Wait for daily quota reset, or use a different Google account/key |
| `503 UNAVAILABLE` | Gemini servers temporarily overloaded | Usually transient — retry in a minute |
| A competitor shows all "Data not found" fields | Either its page failed to fetch, or its content is JS-rendered (marketing sites often are) and `WebBaseLoader` only sees static HTML | Try a different URL for that competitor (e.g. a docs or about page) |

## Extension Ideas

- Add retry-with-backoff before falling back on rate-limit errors
- Add a proper competitor-discovery step (would need a search API)
- Export the briefing to PDF
- Cache fetched pages to avoid re-fetching on repeated runs
