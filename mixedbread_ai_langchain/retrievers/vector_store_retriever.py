from typing import List, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.utils import get_from_dict_or_env
from pydantic import SecretStr, Field, PrivateAttr
from mixedbread import Mixedbread, AsyncMixedbread


class MixedbreadVectorStoreRetriever(BaseRetriever):
    """
    Mixedbread vector store retriever for LangChain.
    
    Provides both chunk-level and file-level search capabilities for Mixedbread AI vector stores.
    """

    vector_store_identifiers: List[str] = Field(
        description="List of vector store IDs to search in"
    )
    top_k: int = Field(default=10, description="Number of top results to return")
    search_type: str = Field(
        default="chunk", description="Search type: 'chunk' or 'file'"
    )
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum relevance score for results"
    )

    _client: Mixedbread = PrivateAttr()
    _async_client: AsyncMixedbread = PrivateAttr()

    def __init__(
        self,
        vector_store_identifiers: List[str],
        api_key: Union[SecretStr, str, None] = None,
        base_url: Optional[str] = None,
        top_k: int = 10,
        search_type: str = "chunk",
        score_threshold: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the Mixedbread vector store retriever.

        Args:
            vector_store_identifiers: List of vector store IDs to search in.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            base_url: Base URL for the API.
            top_k: Number of top results to return.
            search_type: Search type - "chunk" for chunk search, "file" for file search.
            score_threshold: Minimum relevance score for results.
            **kwargs: Additional arguments.
        """
        if not vector_store_identifiers:
            raise ValueError("At least one vector_store_identifier must be provided")

        if search_type not in ["chunk", "file"]:
            raise ValueError("search_type must be 'chunk' or 'file'")

        if api_key is None:
            api_key = get_from_dict_or_env({}, "api_key", "MXBAI_API_KEY")

        if isinstance(api_key, str):
            api_key = SecretStr(api_key)

        super().__init__(
            vector_store_identifiers=vector_store_identifiers,
            top_k=max(1, top_k),
            search_type=search_type,
            score_threshold=score_threshold,
            **kwargs,
        )

        # Initialize clients
        resolved_api_key = api_key.get_secret_value()
        client_kwargs = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Mixedbread(**client_kwargs)
        self._async_client = AsyncMixedbread(**client_kwargs)

    def _convert_results_to_documents(self, search_response) -> List[Document]:
        """Convert search results to LangChain Documents."""
        documents = []

        for item in search_response.data:
            if self.search_type == "chunk":
                # Chunk search results
                page_content = getattr(item, "content", "") or ""
                metadata = {
                    "source": getattr(item, "filename", "unknown"),
                    "score": getattr(item, "score", 0.0),
                }
            else:
                # File search results
                if hasattr(item, "chunks") and item.chunks:
                    # Combine chunk content
                    chunk_texts = [
                        chunk.content for chunk in item.chunks if hasattr(chunk, "content") and chunk.content
                    ]
                    page_content = "\n\n".join(chunk_texts) if chunk_texts else f"File: {getattr(item, 'filename', 'Unknown file')}"
                else:
                    page_content = f"File: {getattr(item, 'filename', 'Unknown file')}"

                metadata = {
                    "source": getattr(item, "filename", "unknown"),
                    "score": getattr(item, "score", 0.0),
                    "file_id": getattr(item, "id", None),
                }

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents for a query (sync version).
        """
        if not query.strip():
            return []

        try:
            search_request = {
                "query": query,
                "vector_store_identifiers": self.vector_store_identifiers,
                "top_k": self.top_k,
            }

            if self.score_threshold is not None:
                search_request["score_threshold"] = self.score_threshold

            if self.search_type == "chunk":
                response = self._client.vector_stores.search(**search_request)
            else:
                response = self._client.vector_stores.files.search(**search_request)

            documents = self._convert_results_to_documents(response)
            return sorted(documents, key=lambda doc: doc.metadata.get("score", 0.0), reverse=True)

        except Exception:
            # Fallback to empty results on error
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents for a query (async version).
        """
        if not query.strip():
            return []

        try:
            search_request = {
                "query": query,
                "vector_store_identifiers": self.vector_store_identifiers,
                "top_k": self.top_k,
            }

            if self.score_threshold is not None:
                search_request["score_threshold"] = self.score_threshold

            if self.search_type == "chunk":
                response = await self._async_client.vector_stores.search(**search_request)
            else:
                response = await self._async_client.vector_stores.files.search(**search_request)

            documents = self._convert_results_to_documents(response)
            return sorted(documents, key=lambda doc: doc.metadata.get("score", 0.0), reverse=True)

        except Exception:
            # Fallback to empty results on error
            return []