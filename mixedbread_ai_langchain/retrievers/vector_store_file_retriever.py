from typing import Any, Dict, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr

from ..common.client import MixedbreadClient


class MixedbreadVectorStoreFileRetriever(BaseRetriever):
    """
    Mixedbread AI Vector Store File retriever for LangChain.
    """

    vector_store_ids: List[str] = Field(
        description="List of vector store IDs to search in"
    )
    top_k: int = Field(default=10, description="Number of top files to return")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum relevance score for results"
    )
    return_metadata: bool = Field(
        default=True, description="Whether to include metadata in results"
    )
    include_chunks: bool = Field(
        default=False, description="Whether to include chunk details in results"
    )
    chunk_limit: Optional[int] = Field(
        default=None, description="Maximum number of chunks to include per file"
    )
    search_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional search options for the API"
    )

    _client: MixedbreadClient = PrivateAttr()

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

        if not vector_store_ids:
            raise ValueError("At least one vector_store_id must be provided")

        super().__init__(
            vector_store_ids=vector_store_ids,
            top_k=max(1, top_k),
            score_threshold=score_threshold,
            return_metadata=return_metadata,
            include_chunks=include_chunks,
            chunk_limit=chunk_limit,
            search_options=search_options or {},
            **kwargs,
        )

        self.search_options.setdefault("return_metadata", self.return_metadata)
        self.search_options.setdefault("return_chunks", self.include_chunks)

        if self.score_threshold is not None:
            self.search_options["score_threshold"] = self.score_threshold

        if self.chunk_limit is not None:
            self.search_options["chunk_limit"] = self.chunk_limit

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
            page_content = ""

            if hasattr(file_result, "chunks") and file_result.chunks:
                chunk_texts = []
                for chunk in file_result.chunks:
                    if hasattr(chunk, "content") and chunk.content:
                        chunk_texts.append(chunk.content)

                if self.chunk_limit and len(chunk_texts) > self.chunk_limit:
                    chunk_texts = chunk_texts[: self.chunk_limit]

                page_content = "\n\n".join(chunk_texts)

            if not page_content:
                page_content = (
                    f"File: {getattr(file_result, 'filename', 'Unknown file')}"
                )

            metadata = {
                "source": getattr(file_result, "filename", "unknown"),
                "file_id": getattr(file_result, "id", None),
                "filename": getattr(file_result, "filename", None),
                "vector_store_id": getattr(file_result, "vector_store_id", None),
                "relevance_score": getattr(file_result, "score", 0.0),
                "status": getattr(file_result, "status", None),
                "usage_bytes": getattr(file_result, "usage_bytes", None),
                "created_at": getattr(file_result, "created_at", None),
                "version": getattr(file_result, "version", None),
                "vector_store_file_retrieval": True,
                "file_based_search": True,
            }

            if hasattr(file_result, "metadata") and file_result.metadata:
                if isinstance(file_result.metadata, dict):
                    metadata["file_metadata"] = file_result.metadata

            if hasattr(file_result, "chunks") and file_result.chunks:
                metadata["total_chunks"] = len(file_result.chunks)
                metadata["chunks_included"] = min(
                    len(file_result.chunks), self.chunk_limit or len(file_result.chunks)
                )

                chunk_details = []
                for i, chunk in enumerate(file_result.chunks):
                    if self.chunk_limit and i >= self.chunk_limit:
                        break

                    chunk_info = {
                        "position": getattr(chunk, "position", i),
                        "score": getattr(chunk, "score", 0.0),
                        "file_id": getattr(chunk, "file_id", None),
                        "vector_store_id": getattr(chunk, "vector_store_id", None),
                    }

                    if hasattr(chunk, "metadata") and chunk.metadata:
                        if isinstance(chunk.metadata, dict):
                            chunk_info["metadata"] = chunk.metadata

                    chunk_details.append(chunk_info)

                metadata["chunk_details"] = chunk_details

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
            return []

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """Get relevant files from vector stores."""
        if not query.strip():
            return []

        documents = self._search_files_in_vector_stores(query)

        documents.sort(
            key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True
        )

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """Async version of _get_relevant_documents."""
        if not query.strip():
            return []

        documents = await self._asearch_files_in_vector_stores(query)

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
