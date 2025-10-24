import logging
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import BaseTool
from typing import Optional, Dict
from qdrant_client import QdrantClient
import threading
logger = logging.getLogger(__name__)


class QdrantClientManager:
    """Singleton manager for QdrantClient instances to prevent database locking issues."""
    _instances: Dict[str, QdrantClient] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_client(cls, persist_dir: str) -> QdrantClient:
        """Get or create a QdrantClient for the given persist directory."""
        with cls._lock:
            if persist_dir not in cls._instances:
                logger.info(f"Creating new QdrantClient for directory: {persist_dir}")
                cls._instances[persist_dir] = QdrantClient(path=persist_dir)
            else:
                logger.debug(f"Reusing existing QdrantClient for directory: {persist_dir}")
            return cls._instances[persist_dir]
    
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
            results = self.retriever.invoke(query)
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
    