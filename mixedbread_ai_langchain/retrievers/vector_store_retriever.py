from typing import Any, Dict, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr

from mixedbread.types import VectorStoreSearchResponse
from ..common.client import MixedbreadClient


class MixedbreadVectorStoreRetriever(BaseRetriever):
    """
    Mixedbread AI Vector Store retriever for LangChain.
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
            return []

    async def _asearch_vector_stores(self, query: str) -> List[Document]:

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
            return []

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

    async def _aget_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:

        if not query.strip():
            return []

        documents = await self._asearch_vector_stores(query)

        documents.sort(
            key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True
        )

        return documents

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
