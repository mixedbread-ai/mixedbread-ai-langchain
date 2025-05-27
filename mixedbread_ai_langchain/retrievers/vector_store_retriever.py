from typing import Any, Dict, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)

from mixedbread.types import VectorStoreSearchResponse
from ..common.client import MixedbreadClient


class MixedbreadVectorStoreRetriever(BaseRetriever):
    """
    Mixedbread AI Vector Store retriever for LangChain.

    This retriever performs semantic search over document chunks stored in
    Mixedbread AI vector stores.
    """

    def __init__(
        self,
        vector_store_ids: List[str],
        api_key: Optional[str] = None,
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
        Initialize the Mixedbread Vector Store retriever.

        Args:
            vector_store_ids: List of vector store IDs to search in
            api_key: API key for Mixedbread AI
            top_k: Number of top results to return
            score_threshold: Minimum relevance score for results
            return_metadata: Whether to include metadata in results
            search_options: Additional search options for the API
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        super().__init__(**kwargs)

        if not vector_store_ids:
            raise ValueError("At least one vector_store_id must be provided")

        self.vector_store_ids = vector_store_ids
        self.top_k = max(1, top_k)
        self.score_threshold = score_threshold
        self.return_metadata = return_metadata
        self.search_options = search_options or {}

        # Set default search options
        self.search_options.setdefault("return_metadata", self.return_metadata)
        if self.score_threshold is not None:
            self.search_options["score_threshold"] = self.score_threshold

        # Initialize the client
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _convert_search_results_to_documents(
        self, search_response: VectorStoreSearchResponse
    ) -> List[Document]:
        """Convert Mixedbread search results to LangChain Documents."""
        documents = []

        for chunk in search_response.data:
            # Extract content
            page_content = chunk.content or ""

            # Build metadata
            metadata = {
                "source": chunk.filename or "unknown",
                "chunk_id": getattr(chunk, "chunk_id", None),
                "relevance_score": chunk.score,
                "vector_store_retrieval": True,
            }

            # Add file metadata if available
            if hasattr(chunk, "file_id") and chunk.file_id:
                metadata["file_id"] = chunk.file_id

            if hasattr(chunk, "filename") and chunk.filename:
                metadata["filename"] = chunk.filename

            # Add chunk position info if available
            if hasattr(chunk, "chunk_index") and chunk.chunk_index is not None:
                metadata["chunk_index"] = chunk.chunk_index

            if hasattr(chunk, "page_number") and chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number

            # Add any additional metadata from the chunk
            if hasattr(chunk, "metadata") and chunk.metadata:
                if isinstance(chunk.metadata, dict):
                    metadata.update(chunk.metadata)

            # Add vector store context
            metadata["vector_store_ids"] = self.vector_store_ids
            metadata["search_top_k"] = self.top_k

            if self.score_threshold is not None:
                metadata["score_threshold"] = self.score_threshold

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    def _search_vector_stores(self, query: str) -> List[Document]:
        """Search the vector stores and return documents."""
        try:
            response: VectorStoreSearchResponse = (
                self._client.client.vector_stores.search(
                    query=query,
                    vector_store_ids=self.vector_store_ids,
                    top_k=self.top_k,
                    search_options=self.search_options,
                )
            )

            return self._convert_search_results_to_documents(response)

        except Exception as e:
            # Log the error but return empty results instead of failing
            return []

    async def _asearch_vector_stores(self, query: str) -> List[Document]:
        """Async version of _search_vector_stores."""
        try:
            response: VectorStoreSearchResponse = (
                await self._client.async_client.vector_stores.search(
                    query=query,
                    vector_store_ids=self.vector_store_ids,
                    top_k=self.top_k,
                    search_options=self.search_options,
                )
            )

            return self._convert_search_results_to_documents(response)

        except Exception as e:
            # Log the error but return empty results instead of failing
            return []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get relevant documents from vector stores."""
        if not query.strip():
            return []

        documents = self._search_vector_stores(query)

        # Sort by relevance score if available
        documents.sort(
            key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True
        )

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of _get_relevant_documents."""
        if not query.strip():
            return []

        documents = await self._asearch_vector_stores(query)

        # Sort by relevance score if available
        documents.sort(
            key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True
        )

        return documents

    def add_vector_store(self, vector_store_id: str) -> None:
        """Add a vector store ID to the search scope."""
        if vector_store_id not in self.vector_store_ids:
            self.vector_store_ids.append(vector_store_id)

    def remove_vector_store(self, vector_store_id: str) -> None:
        """Remove a vector store ID from the search scope."""
        if vector_store_id in self.vector_store_ids:
            self.vector_store_ids.remove(vector_store_id)

        if not self.vector_store_ids:
            raise ValueError("At least one vector_store_id must remain")

    def update_search_options(self, **options: Any) -> None:
        """Update search options."""
        self.search_options.update(options)

        # Update instance variables if they're in the options
        if "score_threshold" in options:
            self.score_threshold = options["score_threshold"]
        if "return_metadata" in options:
            self.return_metadata = options["return_metadata"]
