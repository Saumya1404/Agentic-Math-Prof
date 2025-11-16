import logging
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import BaseTool
from typing import Optional, Dict
from qdrant_client import QdrantClient
import threading
logger = logging.getLogger(__name__)


class QdrantClientManager:
    """Singleton manager for QdrantClient instances to prevent database locking issues.

    Notes:
    - Persist directories are normalized to absolute paths so multiple callers
      using different relative forms don't create multiple underlying QdrantLocal
      instances (which cause file locks on disk-backed storage).
    - When the storage directory is locked by another process, we retry a few
      times with a short backoff to handle races during startup.
    """
    _instances: Dict[str, QdrantClient] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_client(cls, persist_dir: str) -> QdrantClient:
        """Get or create a QdrantClient for the given persist directory.

        persist_dir will be normalized to an absolute path and used as the key.
        """
        from pathlib import Path
        import time

        abs_path = str(Path(persist_dir).resolve())
        with cls._lock:
            if abs_path in cls._instances:
                logger.debug(f"Reusing existing QdrantClient for directory: {abs_path}")
                client = cls._instances[abs_path]
            else:
                logger.info(f"Creating new QdrantClient for directory: {abs_path}")
                # Retry on file-lock/AlreadyLocked for a short period to avoid
                # startup race conditions when multiple processes/threads try to
                # create a QdrantLocal instance for the same folder.
                last_exc = None
                for attempt in range(5):
                    try:
                        client = QdrantClient(path=abs_path)
                        cls._instances[abs_path] = client
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        # Known failure mode on Windows is portalocker.exceptions.AlreadyLocked
                        logger.warning(f"Failed to create QdrantClient for {abs_path} (attempt {attempt+1}/5): {e}")
                        time.sleep(0.25 * (attempt + 1))
                if last_exc is not None:
                    # Final failure: raise a clearer error including the absolute path
                    logger.error(f"Unable to create QdrantClient for {abs_path} after retries: {last_exc}")
                    raise

            # Quick health-check: if client appears closed/unhealthy, recreate it
            client = cls._instances[abs_path]
            try:
                # Many QdrantClient implementations expose get_collections
                client.get_collections()
            except Exception:
                logger.warning(f"Existing QdrantClient for {abs_path} appears closed or unhealthy; recreating.")
                try:
                    client.close()
                except Exception:
                    pass
                client = QdrantClient(path=abs_path)
                cls._instances[abs_path] = client
            return client
    
    @classmethod
    def close_all(cls):
        """Close all QdrantClient instances."""
        with cls._lock:
            for persist_dir, client in cls._instances.items():
                try:
                    client.close()
                    logger.info(f"Closed QdrantClient for directory: {persist_dir}")
                except Exception as e:
                    logger.warning(f"Error closing QdrantClient for {persist_dir}: {e}")
            cls._instances.clear()


class QdrantRetrieverTool(BaseTool):
    """Tool for retrieving information from a Qdrant vector store."""
    name: str
    description: str
    collection_name: str
    persist_dir: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 3
    client: Optional[QdrantClient] = None

 
    embeddings: Optional[HuggingFaceEmbeddings] = None
    vectorstore: Optional[QdrantVectorStore] = None
    retriever: Optional[object] = None

    def __init__(self, **data):
        super().__init__(**data)

        logger.info(f"Loading embedding model '{self.model_name}'...")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        logger.info(f"Connecting to Qdrant collection '{self.collection_name}' at {self.persist_dir}...")
        # Use the singleton manager to get a shared client instance
        self.client = QdrantClientManager.get_client(self.persist_dir)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})


    def _run(self, query: str) -> str:
        logger.debug(f"Retrieving documents for query: {query}")
        try:
            # Ensure underlying client is still the active one from the manager.
            try:
                current_client = QdrantClientManager.get_client(self.persist_dir)
            except Exception:
                current_client = None

            if current_client is not None and getattr(self.vectorstore, "client", None) is not current_client:
                logger.info("Qdrant client changed; recreating vectorstore and retriever")
                try:
                    self.vectorstore = QdrantVectorStore(
                        client=current_client,
                        collection_name=self.collection_name,
                        embedding=self.embeddings
                    )
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                    self.client = current_client
                except Exception:
                    logger.exception("Failed to recreate QdrantVectorStore after client change")

            # Attempt retrieval; if the client was closed unexpectedly, try to recreate once and retry
            try:
                results = self.retriever.invoke(query)
            except Exception as e:
                logger.warning(f"Retrieval failed, attempting one retry after recreating client: {e}")
                try:
                    current_client = QdrantClientManager.get_client(self.persist_dir)
                    self.vectorstore = QdrantVectorStore(
                        client=current_client,
                        collection_name=self.collection_name,
                        embedding=self.embeddings
                    )
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                    self.client = current_client
                    results = self.retriever.invoke(query)
                except Exception as e2:
                    logger.exception(f"Retry retrieval failed: {e2}")
                    raise
            if not results:
                logger.info("No relevant documents found.")
                return "No relevant information found."
            
            formatted_results = "\n\n".join([doc.page_content for doc in results])
            return formatted_results
        except Exception as e:
            logger.exception(f"Error during retrieval: {e}")
            return "An error occurred while retrieving information."
        
    async def _arun(self, query: str) -> str:
        return self._run(query)
    