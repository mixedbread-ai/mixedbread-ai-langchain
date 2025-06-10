from typing import Any, List, Optional, Sequence, Union, Dict
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import SecretStr
from mixedbread.types import RerankResponse
from pydantic import Field, PrivateAttr

from ..common.client import MixedbreadClient
from ..common.mixins import SerializationMixin, AsyncMixin, ErrorHandlingMixin
from ..common.utils import (
    validate_documents,
    create_response_meta,
    create_empty_reranking_response,
    prepare_documents_for_processing,
)
from ..common.logging import get_logger
from langchain_core.callbacks import Callbacks

logger = get_logger(__name__)


class MixedbreadReranker(
    BaseDocumentCompressor, SerializationMixin, AsyncMixin, ErrorHandlingMixin
):
    """
    Document compressor that uses Mixedbread AI for reranking.

    This compressor reranks a list of documents based on their relevance to a query
    using Mixedbread AI's reranking models with enhanced async support, error handling,
    and metadata tracking.
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

    _client: MixedbreadClient = PrivateAttr()

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v2",
        api_key: Union[SecretStr, str, None] = None,
        top_k: int = 3,
        return_input: bool = True,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread reranker.

        Args:
            model: The Mixedbread reranking model to use.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            top_k: Number of top documents to return after reranking.
            return_input: Whether to return the input text in results.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(
            model=model, top_k=max(1, top_k), return_input=return_input, **kwargs
        )

        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _prepare_documents_for_reranking(
        self, documents: Sequence[Document]
    ) -> List[str]:
        """
        Prepare documents for reranking by extracting content.

        Args:
            documents: Sequence of documents to prepare.

        Returns:
            List of text strings ready for reranking.
        """
        return prepare_documents_for_processing(documents)

    def _apply_reranking_results(
        self, documents: Sequence[Document], reranking_response: RerankResponse
    ) -> List[Document]:
        """
        Apply reranking results to documents and return reranked list.

        Args:
            documents: Original sequence of documents.
            reranking_response: Response from the reranking API.

        Returns:
            List of reranked documents with updated metadata.
        """
        if not reranking_response.data:
            logger.warning("No reranking results returned from API")
            return list(documents)

        reranked_docs = []

        # Create response metadata for logging
        meta = create_response_meta(reranking_response, include_reranker_fields=True)
        logger.debug(f"Reranking completed: {meta}")

        for result in reranking_response.data:
            if result.index < len(documents):
                original_doc = documents[result.index]

                # Create new metadata with reranking information
                reranked_metadata = original_doc.metadata.copy()
                reranked_metadata.update(
                    {
                        "rerank_score": result.score,
                        "rerank_index": result.index,
                        "rerank_model": self.model,
                        "original_index": result.index,
                        "rerank_metadata": meta,
                    }
                )

                reranked_doc = Document(
                    page_content=original_doc.page_content, metadata=reranked_metadata
                )
                reranked_docs.append(reranked_doc)
            else:
                logger.warning(
                    f"Rerank result index {result.index} is out of range for {len(documents)} documents"
                )

        return reranked_docs

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """
        Compress documents by reranking them based on relevance to the query.

        Args:
            documents: Sequence of documents to rerank.
            query: The query to rank documents against.

        Returns:
            Sequence of reranked documents, limited to top_k results.
        """
        try:
            validate_documents(list(documents))
        except TypeError as e:
            logger.error(f"Invalid documents provided: {e}")
            return []

        if not documents:
            logger.info("Empty document list provided for reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return documents[: self.top_k]

        try:
            doc_texts = self._prepare_documents_for_reranking(documents)

            response: RerankResponse = self._client.client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_docs = self._apply_reranking_results(documents, response)

            logger.info(
                f"Reranking completed: {len(reranked_docs)} documents returned from {len(documents)} input documents"
            )
            return reranked_docs

        except Exception as e:
            return self._handle_api_error(
                e, "document reranking", documents[: self.top_k]
            )

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """
        Async version of compress_documents.

        Args:
            documents: Sequence of documents to rerank.
            query: The query to rank documents against.

        Returns:
            Sequence of reranked documents, limited to top_k results.
        """
        try:
            validate_documents(list(documents))
        except TypeError as e:
            logger.error(f"Invalid documents provided: {e}")
            return []

        if not documents:
            logger.info("Empty document list provided for async reranking")
            return []

        if not query.strip():
            logger.warning("Empty query provided for async reranking")
            return documents[: self.top_k]

        try:
            doc_texts = self._prepare_documents_for_reranking(documents)

            response: RerankResponse = await self._client.async_client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_docs = self._apply_reranking_results(documents, response)

            logger.info(
                f"Async reranking completed: {len(reranked_docs)} documents returned from {len(documents)} input documents"
            )
            return reranked_docs

        except Exception as e:
            return self._handle_api_error(
                e, "async document reranking", documents[: self.top_k]
            )

    def rerank_with_metadata(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Dict[str, Any]:
        """
        Rerank documents and return both results and metadata.

        Args:
            documents: Sequence of documents to rerank.
            query: The query to rank documents against.

        Returns:
            Dictionary containing reranked documents and metadata.
        """
        try:
            validate_documents(list(documents))
        except TypeError as e:
            logger.error(f"Invalid documents provided: {e}")
            return create_empty_reranking_response(self.model)

        if not documents:
            logger.info("Empty document list provided for reranking with metadata")
            return create_empty_reranking_response(self.model)

        if not query.strip():
            logger.warning("Empty query provided for reranking with metadata")
            empty_response = create_empty_reranking_response(self.model)
            empty_response["documents"] = list(documents[: self.top_k])
            return empty_response

        try:
            doc_texts = self._prepare_documents_for_reranking(documents)

            response: RerankResponse = self._client.client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_docs = self._apply_reranking_results(documents, response)
            meta = create_response_meta(response, include_reranker_fields=True)
            meta["top_k"] = len(reranked_docs)

            return {"documents": reranked_docs, "meta": meta}

        except Exception as e:
            logger.error(f"Error during reranking with metadata: {str(e)}")
            raise

    async def arerank_with_metadata(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Dict[str, Any]:
        """
        Async version of rerank_with_metadata.

        Args:
            documents: Sequence of documents to rerank.
            query: The query to rank documents against.

        Returns:
            Dictionary containing reranked documents and metadata.
        """
        try:
            validate_documents(list(documents))
        except TypeError as e:
            logger.error(f"Invalid documents provided: {e}")
            return create_empty_reranking_response(self.model)

        if not documents:
            logger.info(
                "Empty document list provided for async reranking with metadata"
            )
            return create_empty_reranking_response(self.model)

        if not query.strip():
            logger.warning("Empty query provided for async reranking with metadata")
            empty_response = create_empty_reranking_response(self.model)
            empty_response["documents"] = list(documents[: self.top_k])
            return empty_response

        try:
            doc_texts = self._prepare_documents_for_reranking(documents)

            response: RerankResponse = await self._client.async_client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_docs = self._apply_reranking_results(documents, response)
            meta = create_response_meta(response, include_reranker_fields=True)
            meta["top_k"] = len(reranked_docs)

            return {"documents": reranked_docs, "meta": meta}

        except Exception as e:
            logger.error(f"Error during async reranking with metadata: {str(e)}")
            raise
