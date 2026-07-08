import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
from backend.app.core.logger import logger

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from backend.app.tools.RetrieverTool import QdrantClientManager


FEEDBACK_COLLECTION = "feedback_collection"
FEEDBACK_PERSIST_DIR = "./Data/feedback/qdrant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_lock = threading.Lock()
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[QdrantVectorStore] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading feedback embedding model: {EMBEDDING_MODEL_NAME}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def _get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is not None:
        # Check health: if underlying client was recreated/closed, drop cache
        try:
            existing_client = getattr(_vectorstore, "client", None)
            current_client = QdrantClientManager.get_client(FEEDBACK_PERSIST_DIR)
            if existing_client is not current_client:
                logger.info("Feedback vectorstore client changed; recreating vectorstore")
                _vectorstore = None
            else:
                # quick health check
                try:
                    existing_client.get_collections()
                except Exception:
                    logger.warning("Existing feedback Qdrant client unhealthy; recreating vectorstore")
                    _vectorstore = None
        except Exception:
            _vectorstore = None
        if _vectorstore is not None:
            return _vectorstore

    # Ensure client uses the same persist dir
    client = QdrantClientManager.get_client(FEEDBACK_PERSIST_DIR)
    embeddings = _get_embeddings()
    logger.info(f"Initializing QdrantVectorStore for feedback collection '{FEEDBACK_COLLECTION}' at {FEEDBACK_PERSIST_DIR}")
    # Ensure collection exists; if not, create it with appropriate vector params
    try:
        client.get_collection(collection_name=FEEDBACK_COLLECTION)
    except Exception:
        # compute embedding dim
        try:
            sample_emb = embeddings.embed_documents(["."])[0]
            dim = len(sample_emb)
        except Exception:
            # fallback default dim for all-MiniLM-L6-v2
            dim = 384
        try:
            logger.info(f"Creating Qdrant collection '{FEEDBACK_COLLECTION}' with dim={dim}")
            client.recreate_collection(collection_name=FEEDBACK_COLLECTION, vectors_config=rest_models.VectorParams(size=dim, distance=rest_models.Distance.COSINE))
        except Exception as e:
            logger.exception(f"Failed to create Qdrant collection '{FEEDBACK_COLLECTION}': {e}")

    _vectorstore = QdrantVectorStore(client=client, collection_name=FEEDBACK_COLLECTION, embedding=embeddings)
    return _vectorstore


def add_feedback(record: Dict[str, Any]) -> None:
    """Add a feedback record to Qdrant. The stored text will be a concatenation of human_feedback + initial_response + query."""
    try:
        vs = _get_vectorstore()
        text = "\n---\n".join(
            [
                record.get("human_feedback", ""),
                record.get("initial_response", ""),
                record.get("query", ""),
            ]
        )
        metadata = {
            "query": record.get("query"),
            "initial_response": record.get("initial_response"),
            "human_feedback": record.get("human_feedback"),
            "critic_feedback": record.get("critic_feedback"),
            "context": record.get("context"),
            "refined_response": record.get("refined_response"),
        }
        # Use add_texts provided by LangChain QdrantVectorStore
        vs.add_texts([text], metadatas=[metadata])
        logger.info("Stored feedback record into Qdrant feedback collection.")
    except Exception as e:
        logger.exception(f"Failed to store feedback in Qdrant: {e}")


def get_top_k(text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k similar feedback records as list of dicts with keys: content, metadata."""
    try:
        vs = _get_vectorstore()
        docs = vs.similarity_search(text, k=k)
        results = []
        for d in docs:
            md = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            results.append({"content": d.page_content, "metadata": md})
        return results
    except Exception as e:
        logger.exception(f"Failed to query top-k feedback from Qdrant: {e}")
        return []


def all_feedbacks(limit: int = 100) -> List[Dict[str, Any]]:
    """Return up to `limit` most recent feedback entries (by insertion order)."""
    try:
        vs = _get_vectorstore()
        # LangChain QdrantVectorStore does not expose a direct list-all; fallback to similarity on empty string
        docs = vs.similarity_search("", k=limit)
        results = []
        for d in docs:
            md = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            results.append({"content": d.page_content, "metadata": md})
        return results
    except Exception as e:
        logger.exception(f"Failed to fetch all feedbacks from Qdrant: {e}")
        return []
