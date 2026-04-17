"""
Tools module for ARA.
Unified structured web/news/research search.
"""

import requests
import logging
from typing import Dict, Any
from langchain_core.tools import tool

from src.config import SERPER_API_KEY, NEWSAPI_KEY, TAVILY_API_KEY

logger = logging.getLogger(__name__)


def _fmt(title, url, snippet=None):
    return {"title": title, "url": url, "snippet": snippet or ""}


@tool
def web_search(query: str) -> list[dict]:
    """
    General-purpose web search (Serper).
    Always returns structured list of {title,url,snippet}.
    """
    try:
        if not SERPER_API_KEY:
            return []

        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY},
            json={"q": query, "num": 5},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for o in data.get("organic", [])[:5]:
            out.append(_fmt(o.get("title", ""), o.get("link", ""), o.get("snippet", "")))
        return out
    except Exception as e:
        logger.error(f"web_search error: {e}")
        return []


@tool
def news_search(query: str) -> list[dict]:
    """
    NewsAPI for recent/temporal queries.
    """
    try:
        if not NEWSAPI_KEY:
            return []

        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "sortBy": "publishedAt", "pageSize": 5, "apiKey": NEWSAPI_KEY},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for a in data.get("articles", [])[:5]:
            out.append(_fmt(a.get("title", ""), a.get("url", ""), a.get("description", "")))
        return out
    except Exception as e:
        logger.error(f"news_search error: {e}")
        return []


@tool
def tavily_search(query: str) -> list[dict]:
    """
    Tavily semantic search — excellent for research queries.
    """
    try:
        if not TAVILY_API_KEY:
            return []

        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "n_tokens": 2048},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for r in data.get("results", [])[:5]:
            out.append(_fmt(r.get("title", ""), r.get("url", ""), r.get("content", "")))
        return out
    except Exception as e:
        logger.error(f"tavily_search error: {e}")
        return []


# tool registry used in graph
TOOLS = [web_search, news_search, tavily_search]
