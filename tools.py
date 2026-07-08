"""
tools.py
This is a REAL tool the agent can call.
Unlike the paper search (which only knows what's in paper.pdf),
this function actually reaches out to the internet.
"""

from ddgs import DDGS


def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found on the web for this query."

        combined = "\n\n".join(
            f"Source: {r.get('title', 'Unknown')}\n{r.get('body', '')}"
            for r in results
        )
        return combined

    except Exception as e:
        return f"Web search tool failed: {str(e)}"