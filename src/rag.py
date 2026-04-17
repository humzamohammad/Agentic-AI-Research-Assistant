"""
RAG module with Chroma persistent vector storage.
Per-thread document isolation with fast retrieval.
"""

import os
import tempfile
from typing import Any, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langsmith import traceable
import logging

from src.config import CHROMA_PERSIST_DIR
from src.embeddings import embeddings

logger = logging.getLogger(__name__)


_THREAD_METADATA: Dict[str, dict] = {}


def _get_collection_name(thread_id: str) -> str:
    """Generate a valid Chroma collection name from thread_id."""
    return f"thread_{thread_id.replace('-', '_')}"


@traceable(name="ingest_pdf")
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Index PDF into Chroma vector store for the thread.
    Persists embeddings to disk for fast retrieval.
    """
    if not file_bytes:
        return {"success": False, "error": "No file content received"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            return {"success": False, "error": "PDF is empty or unreadable"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        collection_name = _get_collection_name(thread_id)

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR
        )

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "pages": len(docs),
            "chunks": len(chunks),
        }

        logger.info(f"Indexed {len(chunks)} chunks for thread {thread_id}")
        return {
            "success": True,
            "filename": filename or os.path.basename(temp_path),
            "pages": len(docs),
            "chunks": len(chunks),
        }
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _get_vectorstore(thread_id: str) -> Optional[Chroma]:
    """Get or create Chroma vectorstore for thread."""
    try:
        collection_name = _get_collection_name(thread_id)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        if vectorstore._collection.count() > 0:
            return vectorstore
    except Exception:
        pass
    return None


@tool
def rag_search(query: str, thread_id: str) -> str:
    """
    Search the uploaded PDF document for relevant information.

    Args:
        query: What to search for in the document
        thread_id: The conversation thread ID

    Returns:
        Relevant text excerpts from the document
    """
    vectorstore = _get_vectorstore(thread_id)
    if vectorstore is None:
        return "No document has been uploaded for this conversation. Please upload a PDF first."

    try:
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            return "No relevant information found in the document."

        results = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            results.append(f"[Page {page}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return f"Document search failed: {str(e)}"


def thread_has_document(thread_id: str) -> bool:
    """Check if thread has an uploaded document."""
    if str(thread_id) in _THREAD_METADATA:
        return True
    return _get_vectorstore(thread_id) is not None


def thread_document_metadata(thread_id: str) -> dict:
    """Get document metadata for a thread."""
    return _THREAD_METADATA.get(str(thread_id), {})
