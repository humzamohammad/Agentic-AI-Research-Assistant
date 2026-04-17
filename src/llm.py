"""
Groq LLM factory module.

Provides a single factory function to create ChatGroq instances with
automatic fallback across API keys and model tiers.

Models:
    PRIMARY: llama-3.3-70b-versatile (high quality, may have rate limits)
    FALLBACK: llama-3.1-8b-instant (faster, lower quality)

Fallback Strategy:
    1. Try PRIMARY model with GROQ_API_KEY
    2. Try FALLBACK model with GROQ_API_KEY
    3. Try PRIMARY model with GROQ_API_KEY_BACKUP
    4. Try FALLBACK model with GROQ_API_KEY_BACKUP
    5. Raise RuntimeError if all attempts fail
"""

import logging
from langchain_groq import ChatGroq
from langsmith import traceable

from src.config import GROQ_API_KEY, GROQ_API_KEY_BACKUP

logger = logging.getLogger(__name__)

PRIMARY = "llama-3.3-70b-versatile"
FALLBACK = "llama-3.1-8b-instant"


@traceable(name="create_llm")
def create_llm(temperature: float = 0.3, model: str | None = None) -> ChatGroq:
    """
    Create a Groq chat model with automatic key and model fallback.

    Attempts to initialize ChatGroq using the primary API key and model first.
    If initialization fails, tries fallback model, then backup API key.

    Args:
        temperature (float): Sampling temperature for generation. Lower values
            produce more deterministic outputs. Default is 0.3.
        model (str | None): Specific model to use. If None, uses PRIMARY model
            (llama-3.3-70b-versatile) with FALLBACK as backup.

    Returns:
        ChatGroq: Configured Groq chat model ready for invocation.

    Raises:
        RuntimeError: If no working API key + model combination is found.

    Side Effects:
        Logs successful initialization or warning on each failed attempt.
    """
    mdl = model or PRIMARY

    for key in (GROQ_API_KEY, GROQ_API_KEY_BACKUP):
        if not key:
            continue

        for candidate in (mdl, FALLBACK):
            try:
                llm = ChatGroq(
                    model=candidate,
                    temperature=temperature,
                    groq_api_key=key,
                    max_retries=2,
                )
                logger.info(f"Groq: initialized model {candidate}")
                return llm
            except Exception as e:
                logger.warning(f"Groq init failed for {candidate}: {e}")

    raise RuntimeError("No Groq model available.")
