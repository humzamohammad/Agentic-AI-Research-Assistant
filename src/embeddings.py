"""
Embedding backend for RAG using HuggingFace Inference API.

Provides a LangChain-compatible Embeddings class that calls HuggingFace's
feature_extraction endpoint. Handles the quirks of HF API responses:
- Tensor/ndarray/list return types
- Nested shape variations ([[[vec]]] vs [[vec]])
- 504 timeouts with retry logic

Model: BAAI/bge-small-en-v1.5 (384 dimensions)

Usage:
    from src.embeddings import embeddings
    vectors = embeddings.embed_documents(["text1", "text2"])
    query_vec = embeddings.embed_query("search query")
"""

import time
import logging
from typing import List
from langsmith import traceable
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

from src.config import HUGGINGFACE_API_KEY

logger = logging.getLogger(__name__)


class HFInferenceEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using HuggingFace Inference API.

    Implements batched embedding with retry logic to handle HF API timeouts.
    Normalizes inconsistent response shapes from feature_extraction endpoint.

    Attributes:
        client: HuggingFace InferenceClient instance.
        model: Model ID for embeddings (default: BAAI/bge-small-en-v1.5).
        batch_size: Number of texts to embed per API call (default: 16).
        max_retries: Retry attempts on failure (default: 3).
        dim: Embedding dimension (384 for bge-small).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 16,
        max_retries: int = 3,
    ):
        """
        Initialize the HuggingFace embedding client.

        Args:
            model (str): HuggingFace model ID for embeddings.
            batch_size (int): Texts per API call to avoid timeouts.
            max_retries (int): Number of retry attempts on API failure.
        """
        self.client = InferenceClient(token=HUGGINGFACE_API_KEY)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dim = 384

    def _convert_to_list(self, raw):
        """
        Normalize HuggingFace API response to List[List[float]].

        Handles various return shapes from feature_extraction:
        - numpy arrays (have .tolist())
        - nested lists like [[[vec]]] or [[vec]]
        - empty responses

        Args:
            raw: Raw response from HF feature_extraction endpoint.

        Returns:
            List[List[float]]: Normalized embeddings, one per input text.
        """
        if hasattr(raw, "tolist"):
            arr = raw.tolist()
        else:
            arr = raw

        if not isinstance(arr, list) or len(arr) == 0:
            return []

        first = arr[0]

        # Handle triple-nested shape [[[vec]]]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
            arr = [x for x in first]

        return arr

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.

        Calls HuggingFace feature_extraction with exponential backoff on failure.
        Returns zero vectors if all retries are exhausted.

        Args:
            texts (List[str]): Texts to embed (should be <= batch_size).

        Returns:
            List[List[float]]: Embedding vectors, one per input text.
                Returns zero vectors on complete failure.
        """
        for attempt in range(self.max_retries):
            try:
                raw = self.client.feature_extraction(texts, model=self.model)
                arr = self._convert_to_list(raw)

                if isinstance(arr, list) and len(arr) > 0:
                    return arr

                logger.warning(f"Empty embedding batch (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Embedding retry {attempt + 1}: {e}")
                time.sleep(0.4 * (attempt + 1))

        logger.error("Embedding failed after retries; returning zero vectors.")
        return [[0.0] * self.dim for _ in texts]

    @traceable(name="embed_documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents in batches.

        Splits input into batch_size chunks and embeds each via _embed_batch.
        Results are concatenated in order.

        Args:
            texts (List[str]): Documents to embed.

        Returns:
            List[List[float]]: Embedding vectors, one per document.
        """
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embs.extend(self._embed_batch(batch))
        return all_embs

    @traceable(name="embed_query")
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding vector (384 dimensions).
                Returns zero vector on failure.
        """
        result = self._embed_batch([text])
        if result and len(result) > 0:
            return result[0]
        return [0.0] * self.dim


embeddings = HFInferenceEmbeddings()
