from typing import Any, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from mixedbread.types import RerankResponse
from pydantic import Field, PrivateAttr
from ..common.client import MixedbreadClient


class MixedbreadReranker(BaseDocumentCompressor):
    """
    Document compressor that uses Mixedbread AI for reranking.

    This compressor reranks a list of documents based on their relevance to a query
    using Mixedbread AI's reranking models.
    """

    model: str = Field(
        default="mixedbread-ai/mxbai-rerank-large-v2",
        description="The Mixedbread reranking model to use",
    )
    top_k: int = Field(
        default=3, description="Number of top documents to return after reranking"
    )
    return_input: bool = Field(
        default=True, description="Whether to return the input text in results"
    )
    rank_fields: List[str] = Field(
        default_factory=list,
        description="Specific metadata fields to include in ranking text",
    )

    _client: MixedbreadClient = PrivateAttr()

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        api_key: Optional[str] = None,
        top_k: int = 3,
        return_input: bool = True,
        rank_fields: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread reranker.

        Args:
            model: The Mixedbread reranking model to use
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var)
            top_k: Number of top documents to return after reranking
            return_input: Whether to return the input text in results
            rank_fields: Specific metadata fields to include in ranking text
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        # Initialize Pydantic fields
        super().__init__(
            model=model,
            top_k=max(1, top_k),
            return_input=return_input,
            rank_fields=rank_fields or [],
            **kwargs,
        )

        # Initialize the private client
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _prepare_documents_for_reranking(
        self, documents: Sequence[Document]
    ) -> List[str]:
        """Prepare documents for reranking by extracting text content."""
        prepared_texts = []

        for doc in documents:
            content = doc.page_content or ""

            if self.rank_fields:
                meta_content = []
                for field in self.rank_fields:
                    field_value = doc.metadata.get(field)
                    if field_value is not None:
                        meta_content.append(str(field_value))

                if meta_content:
                    # Combine metadata with content
                    combined_content = " ".join(meta_content)
                    if content:
                        content = f"{combined_content} {content}"
                    else:
                        content = combined_content

            prepared_texts.append(content)

        return prepared_texts

    def _apply_reranking_results(
        self, documents: Sequence[Document], reranking_response: RerankResponse
    ) -> List[Document]:
        """Apply reranking results to documents and return reranked list."""
        if not reranking_response.data:
            return list(documents)

        reranked_docs = []

        for result in reranking_response.data:
            if result.index < len(documents):
                original_doc = documents[result.index]

                reranked_metadata = original_doc.metadata.copy()
                reranked_metadata.update(
                    {
                        "rerank_score": result.score,
                        "rerank_index": result.index,
                        "rerank_model": self.model,
                        "original_index": result.index,
                    }
                )

                reranked_doc = Document(
                    page_content=original_doc.page_content, metadata=reranked_metadata
                )
                reranked_docs.append(reranked_doc)

        return reranked_docs

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using Mixedbread AI.

        Args:
            documents: A sequence of documents to compress/rerank
            query: The query to use for reranking the documents
            callbacks: Optional callbacks to run during compression

        Returns:
            A sequence of reranked documents, limited to top_k
        """
        if not documents:
            return []

        if not query.strip():
            return documents[: self.top_k]

        try:
            # Prepare document texts for reranking
            doc_texts = self._prepare_documents_for_reranking(documents)

            # Call the reranking API
            response: RerankResponse = self._client.client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            # Apply reranking results
            reranked_docs = self._apply_reranking_results(documents, response)

            return reranked_docs

        except Exception as e:
            # Fallback: return original documents (limited to top_k) if reranking fails
            return documents[: self.top_k]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Async version of compress_documents.

        Args:
            documents: A sequence of documents to compress/rerank
            query: The query to use for reranking the documents
            callbacks: Optional callbacks to run during compression

        Returns:
            A sequence of reranked documents, limited to top_k
        """
        if not documents:
            return []

        if not query.strip():
            return documents[: self.top_k]

        try:
            # Prepare document texts for reranking
            doc_texts = self._prepare_documents_for_reranking(documents)

            # Call the reranking API asynchronously
            response: RerankResponse = await self._client.async_client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            # Apply reranking results
            reranked_docs = self._apply_reranking_results(documents, response)

            return reranked_docs

        except Exception as e:
            # Fallback: return original documents (limited to top_k) if reranking fails
            return documents[: self.top_k]

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters."""
        if "top_k" in kwargs:
            self.top_k = max(1, kwargs["top_k"])
        if "return_input" in kwargs:
            self.return_input = kwargs["return_input"]
        if "rank_fields" in kwargs:
            self.rank_fields = kwargs["rank_fields"] or []
