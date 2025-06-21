from typing import Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.utils import get_from_dict_or_env
from pydantic import SecretStr, Field, PrivateAttr
from mixedbread import Mixedbread, AsyncMixedbread


class MixedbreadReranker(BaseDocumentCompressor):
    """
    Mixedbread reranker integration for LangChain.

    Implementation that reranks documents based on relevance to a query
    using the Mixedbread API while following LangChain's BaseDocumentCompressor interface.
    """

    model: str = Field(default="mixedbread-ai/mxbai-rerank-large-v2")
    top_k: int = Field(default=3)
    return_input: bool = Field(default=True)

    # Private attributes for clients
    _client: Mixedbread = PrivateAttr()
    _async_client: AsyncMixedbread = PrivateAttr()

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v2",
        api_key: Union[SecretStr, str, None] = None,
        base_url: Optional[str] = None,
        top_k: int = 3,
        return_input: bool = True,
        **kwargs,
    ):
        """
        Initialize the Mixedbread reranker.

        Args:
            model: The Mixedbread reranking model to use.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            base_url: Base URL for the API.
            top_k: Number of top documents to return after reranking.
            return_input: Whether to return the input text in results.
            **kwargs: Additional arguments.
        """
        if api_key is None:
            api_key = get_from_dict_or_env({}, "api_key", "MXBAI_API_KEY")

        if isinstance(api_key, str):
            api_key = SecretStr(api_key)

        # Initialize Pydantic model
        super().__init__(
            model=model, top_k=max(1, top_k), return_input=return_input, **kwargs
        )

        # Initialize clients
        resolved_api_key = api_key.get_secret_value()
        client_kwargs = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Mixedbread(**client_kwargs)
        self._async_client = AsyncMixedbread(**client_kwargs)

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
        if not documents:
            return []

        if not query.strip():
            return list(documents[: self.top_k])

        # Extract document texts
        doc_texts = [doc.page_content or "" for doc in documents]

        try:
            response = self._client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            if not response.data:
                return list(documents[: self.top_k])

            # Create reranked documents with scores
            reranked_docs = []
            for result in response.data:
                if result.index < len(documents):
                    original_doc = documents[result.index]

                    # Add minimal reranking metadata
                    reranked_metadata = original_doc.metadata.copy()
                    reranked_metadata.update(
                        {
                            "rerank_score": result.score,
                            "rerank_index": result.index,
                        }
                    )

                    reranked_doc = Document(
                        page_content=original_doc.page_content,
                        metadata=reranked_metadata,
                    )
                    reranked_docs.append(reranked_doc)

            return reranked_docs

        except Exception:
            # Fallback to original order on error
            return list(documents[: self.top_k])

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
        if not documents:
            return []

        if not query.strip():
            return list(documents[: self.top_k])

        # Extract document texts
        doc_texts = [doc.page_content or "" for doc in documents]

        try:
            response = await self._async_client.rerank(
                model=self.model,
                query=query,
                input=doc_texts,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            if not response.data:
                return list(documents[: self.top_k])

            # Create reranked documents with scores
            reranked_docs = []
            for result in response.data:
                if result.index < len(documents):
                    original_doc = documents[result.index]

                    # Add minimal reranking metadata
                    reranked_metadata = original_doc.metadata.copy()
                    reranked_metadata.update(
                        {
                            "rerank_score": result.score,
                            "rerank_index": result.index,
                        }
                    )

                    reranked_doc = Document(
                        page_content=original_doc.page_content,
                        metadata=reranked_metadata,
                    )
                    reranked_docs.append(reranked_doc)

            return reranked_docs

        except Exception:
            # Fallback to original order on error
            return list(documents[: self.top_k])
