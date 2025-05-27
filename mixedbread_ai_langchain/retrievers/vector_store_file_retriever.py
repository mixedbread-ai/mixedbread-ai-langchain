from typing import Any, Dict, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)

from ..common.client import MixedbreadClient


class MixedbreadVectorStoreFileRetriever(BaseRetriever):
    """
    Mixedbread AI Vector Store File retriever for LangChain.

    This retriever performs semantic search over entire files stored in
    Mixedbread AI vector stores, with optional chunk details.
    """

    def __init__(
        self,
        vector_store_ids: List[str],
        api_key: Optional[str] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        return_metadata: bool = True,
        include_chunks: bool = False,
        chunk_limit: Optional[int] = None,
        search_options: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread Vector Store File retriever.

        Args:
            vector_store_ids: List of vector store IDs to search in
            api_key: API key for Mixedbread AI
            top_k: Number of top files to return
            score_threshold: Minimum relevance score for results
            return_metadata: Whether to include metadata in results
            include_chunks: Whether to include chunk details in results
            chunk_limit: Maximum number of chunks to include per file
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
        self.include_chunks = include_chunks
        self.chunk_limit = chunk_limit
        self.search_options = search_options or {}

        # Set default search options
        self.search_options.setdefault("return_metadata", self.return_metadata)
        self.search_options.setdefault("return_chunks", self.include_chunks)

        if self.score_threshold is not None:
            self.search_options["score_threshold"] = self.score_threshold

        if self.chunk_limit is not None:
            self.search_options["chunk_limit"] = self.chunk_limit

        # Initialize the client
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _convert_file_results_to_documents(self, search_response) -> List[Document]:
        """Convert Mixedbread file search results to LangChain Documents."""
        documents = []

        for file_result in search_response.data:
            # For file-based retrieval, we create one document per file
            # The page_content will be the file summary or first chunk
            page_content = ""

            # Try to get file content from chunks if available
            if hasattr(file_result, "chunks") and file_result.chunks:
                # Combine chunks into page content
                chunk_texts = []
                for chunk in file_result.chunks:
                    if hasattr(chunk, "content") and chunk.content:
                        chunk_texts.append(chunk.content)

                # Limit chunks if specified
                if self.chunk_limit and len(chunk_texts) > self.chunk_limit:
                    chunk_texts = chunk_texts[: self.chunk_limit]

                page_content = "\n\n".join(chunk_texts)

            # If no chunks, use file name as content placeholder
            if not page_content:
                page_content = (
                    f"File: {getattr(file_result, 'filename', 'Unknown file')}"
                )

            # Build metadata
            metadata = {
                "source": getattr(file_result, "filename", "unknown"),
                "file_id": getattr(file_result, "file_id", None),
                "relevance_score": getattr(file_result, "score", 0.0),
                "vector_store_file_retrieval": True,
                "file_based_search": True,
            }

            # Add file metadata if available
            if hasattr(file_result, "filename") and file_result.filename:
                metadata["filename"] = file_result.filename

            if hasattr(file_result, "file_size") and file_result.file_size:
                metadata["file_size"] = file_result.file_size

            if hasattr(file_result, "file_type") and file_result.file_type:
                metadata["file_type"] = file_result.file_type

            if hasattr(file_result, "upload_date") and file_result.upload_date:
                metadata["upload_date"] = file_result.upload_date

            # Add chunk information if included
            if hasattr(file_result, "chunks") and file_result.chunks:
                metadata["total_chunks"] = len(file_result.chunks)
                metadata["chunks_included"] = min(
                    len(file_result.chunks), self.chunk_limit or len(file_result.chunks)
                )

                # Add chunk metadata
                chunk_metadata = []
                for i, chunk in enumerate(file_result.chunks):
                    if self.chunk_limit and i >= self.chunk_limit:
                        break

                    chunk_info = {
                        "chunk_index": getattr(chunk, "chunk_index", i),
                        "chunk_score": getattr(chunk, "score", 0.0),
                    }

                    if hasattr(chunk, "page_number") and chunk.page_number is not None:
                        chunk_info["page_number"] = chunk.page_number

                    chunk_metadata.append(chunk_info)

                metadata["chunk_details"] = chunk_metadata

            # Add any additional file metadata
            if hasattr(file_result, "metadata") and file_result.metadata:
                if isinstance(file_result.metadata, dict):
                    metadata.update(file_result.metadata)

            # Add vector store context
            metadata["vector_store_ids"] = self.vector_store_ids
            metadata["search_top_k"] = self.top_k
            metadata["include_chunks"] = self.include_chunks

            if self.score_threshold is not None:
                metadata["score_threshold"] = self.score_threshold

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    def _search_files_in_vector_stores(self, query: str) -> List[Document]:
        """Search files in vector stores and return documents."""
        try:
            response = self._client.client.vector_stores.files.search(
                query=query,
                vector_store_ids=self.vector_store_ids,
                top_k=self.top_k,
                search_options=self.search_options,
            )

            return self._convert_file_results_to_documents(response)

        except Exception as e:
            # Log the error but return empty results instead of failing
            return []

    async def _asearch_files_in_vector_stores(self, query: str) -> List[Document]:
        """Async version of _search_files_in_vector_stores."""
        try:
            response = await self._client.async_client.vector_stores.files.search(
                query=query,
                vector_store_ids=self.vector_store_ids,
                top_k=self.top_k,
                search_options=self.search_options,
            )

            return self._convert_file_results_to_documents(response)

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
        """Get relevant files from vector stores."""
        if not query.strip():
            return []

        documents = self._search_files_in_vector_stores(query)

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

        documents = await self._asearch_files_in_vector_stores(query)

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
        if "return_chunks" in options:
            self.include_chunks = options["return_chunks"]
        if "chunk_limit" in options:
            self.chunk_limit = options["chunk_limit"]

    def set_chunk_inclusion(
        self, include_chunks: bool, chunk_limit: Optional[int] = None
    ) -> None:
        """Configure chunk inclusion in file results."""
        self.include_chunks = include_chunks
        self.chunk_limit = chunk_limit

        self.search_options["return_chunks"] = include_chunks
        if chunk_limit is not None:
            self.search_options["chunk_limit"] = chunk_limit
