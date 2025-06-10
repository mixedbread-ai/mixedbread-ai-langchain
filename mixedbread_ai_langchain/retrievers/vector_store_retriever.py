from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import SecretStr, Field, PrivateAttr

from mixedbread.types import VectorStoreSearchResponse
from ..common.client import MixedbreadClient
from ..common.mixins import SerializationMixin, AsyncMixin, ErrorHandlingMixin
from ..common.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mixedbread import AsyncMixedbread as AsyncMixedbreadSDKClient
else:
    AsyncMixedbreadSDKClient = Any


class MixedbreadVectorStoreRetriever(
    BaseRetriever, SerializationMixin, AsyncMixin, ErrorHandlingMixin
):
    """
    Mixedbread AI Vector Store retriever for LangChain.

    This retriever searches through Mixedbread AI vector stores with enhanced
    async support, error handling, and direct client access.

    Usage:
        # Sync operations
        retriever = MixedbreadVectorStoreRetriever(vector_store_ids=["store_id"])
        docs = retriever.get_relevant_documents("query")

        # Async operations (direct client access)
        results = await retriever.aclient.vector_stores.search(
            query="query", vector_store_ids=["store_id"], top_k=5
        )
    """

    vector_store_ids: List[str] = Field(
        description="List of vector store IDs to search in"
    )
    top_k: int = Field(default=10, description="Number of top results to return")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum relevance score for results"
    )
    return_metadata: bool = Field(
        default=True, description="Whether to include metadata in results"
    )
    search_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional search options for the API"
    )

    _client: MixedbreadClient = PrivateAttr()
    # Note: async client is accessed via self._client.async_client

    def __init__(
        self,
        vector_store_ids: List[str],
        api_key: Union[SecretStr, str, None] = None,
        sync_client: Optional[MixedbreadClient] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        return_metadata: bool = True,
        search_options: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread vector store retriever.

        Args:
            vector_store_ids: List of vector store IDs to search in.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            sync_client: Pre-configured sync client (optional).
            async_client: Pre-configured async client (optional).
            top_k: Number of top results to return.
            score_threshold: Minimum relevance score for results.
            return_metadata: Whether to include metadata in results.
            search_options: Additional search options for the API.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            **kwargs: Additional arguments passed to parent classes.
        """
        if not vector_store_ids:
            raise ValueError("At least one vector_store_id must be provided")

        super().__init__(
            vector_store_ids=vector_store_ids,
            top_k=max(1, top_k),
            score_threshold=score_threshold,
            return_metadata=return_metadata,
            search_options=search_options or {},
            **kwargs,
        )

        self.search_options.setdefault("return_metadata", self.return_metadata)
        if self.score_threshold is not None:
            self.search_options["score_threshold"] = self.score_threshold

        # Use provided clients or create new ones
        if sync_client:
            self._client = sync_client
        else:
            self._client = MixedbreadClient(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        # Use unified client approach - async client is available via MixedbreadClient
        # No separate async client management needed

    @property
    def aclient(self) -> AsyncMixedbreadSDKClient:
        """Get the async client for direct API access."""
        return self._client.async_client

    def _convert_search_results_to_documents(
        self, search_response: VectorStoreSearchResponse
    ) -> List[Document]:
        """Convert Mixedbread search results to LangChain Documents."""
        documents = []

        for chunk in search_response.data:
            page_content = getattr(chunk, "content", "") or ""

            metadata = {
                "source": getattr(chunk, "filename", "unknown"),
                "position": getattr(chunk, "position", None),
                "score": getattr(chunk, "score", 0.0),
                "file_id": getattr(chunk, "file_id", None),
                "filename": getattr(chunk, "filename", None),
                "vector_store_id": getattr(chunk, "vector_store_id", None),
                "value": getattr(chunk, "value", None),
                "relevance_score": getattr(chunk, "score", 0.0),
                "vector_store_retrieval": True,
            }

            if hasattr(chunk, "metadata") and chunk.metadata:
                if isinstance(chunk.metadata, dict):
                    metadata["chunk_metadata"] = chunk.metadata

            metadata["vector_store_ids"] = self.vector_store_ids
            metadata["search_top_k"] = self.top_k

            if self.score_threshold is not None:
                metadata["score_threshold"] = self.score_threshold

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    def _search_vector_stores(self, query: str) -> List[Document]:
        """
        Search vector stores and return documents.

        Args:
            query: Query string to search for.

        Returns:
            List of documents from search results.
        """
        try:
            logger.debug(
                f"Searching vector stores {self.vector_store_ids} with query: {query[:50]}..."
            )

            # Prepare request body for the new API structure
            search_request = {
                "query": query,
                "vector_store_ids": self.vector_store_ids,
                "top_k": self.top_k,
            }

            # Add optional parameters
            if self.score_threshold is not None:
                search_request["score_threshold"] = self.score_threshold
            if self.return_metadata:
                search_request["return_metadata"] = self.return_metadata

            # Add any additional search options
            search_request.update(self.search_options)

            response: VectorStoreSearchResponse = (
                self._client.client.vector_stores.search(**search_request)
            )

            documents = self._convert_search_results_to_documents(response)
            logger.info(f"Vector store search returned {len(documents)} documents")
            return documents

        except Exception as e:
            return self._handle_api_error(e, "vector store search", [])

    def search_async(self, query: str):
        """
        Convenience method that returns an awaitable for async search.
        For direct async operations, use retriever.aclient.vector_stores.search() instead.

        Args:
            query: Query string to search for.

        Returns:
            Awaitable that resolves to search response from the API.
        """
        # Async client is always available through MixedbreadClient
        async_client = self._client.async_client

        search_request = {
            "query": query,
            "vector_store_ids": self.vector_store_ids,
            "top_k": self.top_k,
        }

        # Add optional parameters
        if self.score_threshold is not None:
            search_request["score_threshold"] = self.score_threshold
        if self.return_metadata:
            search_request["return_metadata"] = self.return_metadata

        # Add any additional search options
        search_request.update(self.search_options)

        return async_client.vector_stores.search(**search_request)

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:

        if not query.strip():
            return []

        documents = self._search_vector_stores(query)

        documents.sort(
            key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True
        )

        return documents

    # Note: For async operations, use retriever.search_async() or retriever.aclient.vector_stores.search() directly

    def add_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id not in self.vector_store_ids:
            self.vector_store_ids.append(vector_store_id)

    def remove_vector_store(self, vector_store_id: str) -> None:

        if vector_store_id in self.vector_store_ids:
            self.vector_store_ids.remove(vector_store_id)

        if not self.vector_store_ids:
            raise ValueError("At least one vector_store_id must remain")

    def update_search_options(self, **options: Any) -> None:
        self.search_options.update(options)

        if "score_threshold" in options:
            self.score_threshold = options["score_threshold"]
        if "return_metadata" in options:
            self.return_metadata = options["return_metadata"]
