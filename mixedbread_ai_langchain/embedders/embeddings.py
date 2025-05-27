import asyncio
from typing import Any, List, Optional, Union

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from mixedbread.types import EmbeddingCreateResponse
from .types import MixedbreadEmbeddingType, EmbeddingVector
from ..common.client import MixedbreadClient


def _extract_embedding_from_item(item: Any) -> EmbeddingVector:
    """Extracts a single embedding vector from an SDK response item."""
    if hasattr(item, "embedding"):
        return item.embedding
    if hasattr(item, "float") and item.float:
        return item.float
    if hasattr(item, "int8") and item.int8:
        return item.int8
    if hasattr(item, "base64") and item.base64:
        return item.base64

    # Fallback: try to access attributes dynamically
    for attr_name in ["embedding", "float", "int8", "base64"]:
        if hasattr(item, attr_name):
            value = getattr(item, attr_name)
            if value is not None:
                return value

    return []


class MixedbreadEmbeddings(Embeddings):
    """
    Mixedbread AI embeddings integration for LangChain.

    This class provides both text and document embedding capabilities,
    integrating with LangChain's standard Embeddings interface.

    Example:
        .. code-block:: python

            from mixedbread_ai_langchain import MixedbreadEmbeddings

            embeddings = MixedbreadEmbeddings(
                model="mixedbread-ai/mxbai-embed-large-v1",
                api_key="your-api-key"
            )

            # Embed a single query
            query_embedding = embeddings.embed_query("What is AI?")

            # Embed multiple documents
            docs = ["Document 1", "Document 2"]
            doc_embeddings = embeddings.embed_documents(docs)
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        api_key: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        normalized: bool = True,
        encoding_format: Union[
            str, MixedbreadEmbeddingType
        ] = MixedbreadEmbeddingType.FLOAT,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        batch_size: int = 128,
        embedding_separator: str = "\n",
        meta_fields_to_embed: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread embeddings.

        Args:
            model: The Mixedbread model to use for embeddings
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var)
            prefix: Prefix to add to each text before embedding
            suffix: Suffix to add to each text before embedding
            normalized: Whether to normalize the embeddings
            encoding_format: Format for the embeddings (float, int8, etc.)
            dimensions: Target dimensions for the embeddings
            prompt: Optional prompt to use for embeddings
            batch_size: Batch size for processing multiple texts
            embedding_separator: Separator for joining metadata fields
            meta_fields_to_embed: List of metadata fields to include in embedding
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.normalized = normalized

        if isinstance(encoding_format, str):
            self.encoding_format = MixedbreadEmbeddingType.from_str(encoding_format)
        else:
            self.encoding_format = encoding_format

        self.dimensions = dimensions
        self.prompt = prompt
        self.batch_size = min(max(1, batch_size), 256)  # Clamp between 1 and 256
        self.embedding_separator = embedding_separator
        self.meta_fields_to_embed = meta_fields_to_embed or []

        # Initialize the client
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Prepare texts with prefix and suffix."""
        return [f"{self.prefix}{text}{self.suffix}" for text in texts]

    def _prepare_documents_for_embedding(self, documents: List[Document]) -> List[str]:
        """Prepare documents for embedding by combining content with metadata."""
        texts = []
        for doc in documents:
            meta_content = self.embedding_separator.join(
                str(doc.metadata[key])
                for key in self.meta_fields_to_embed
                if doc.metadata.get(key) is not None
            )
            content_to_embed = doc.page_content or ""

            if meta_content and content_to_embed:
                full_text = meta_content + self.embedding_separator + content_to_embed
            elif meta_content:
                full_text = meta_content
            else:
                full_text = content_to_embed

            texts.append(f"{self.prefix}{full_text}{self.suffix}")
        return texts

    def _get_embeddings(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[EmbeddingVector]:
        """Get embeddings for a list of texts."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            response: EmbeddingCreateResponse = self._client.client.embed(
                model=self.model,
                input=batch_texts,
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=prompt or self.prompt,
            )

            batch_embeddings = (
                [_extract_embedding_from_item(item) for item in response.data]
                if response.data
                else []
            )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _aget_embeddings(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[EmbeddingVector]:
        """Async version of _get_embeddings."""
        if not texts:
            return []

        all_embeddings = []

        # Process batches concurrently
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            tasks.append(self._aget_batch_embeddings(batch_texts, prompt))

        results = await asyncio.gather(*tasks)
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _aget_batch_embeddings(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[EmbeddingVector]:
        """Get embeddings for a single batch asynchronously."""
        response: EmbeddingCreateResponse = await self._client.async_client.embed(
            model=self.model,
            input=texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        return (
            [_extract_embedding_from_item(item) for item in response.data]
            if response.data
            else []
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        prepared_texts = self._prepare_texts([text])
        embeddings = self._get_embeddings(prepared_texts)
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings")

        if not texts:
            return []

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("All items in the input list must be strings")

        prepared_texts = self._prepare_texts(texts)
        return self._get_embeddings(prepared_texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        prepared_texts = self._prepare_texts([text])
        embeddings = await self._aget_embeddings(prepared_texts)
        return embeddings[0] if embeddings else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings")

        if not texts:
            return []

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("All items in the input list must be strings")

        prepared_texts = self._prepare_texts(texts)
        return await self._aget_embeddings(prepared_texts)

    def embed_langchain_documents(self, documents: List[Document]) -> List[Document]:
        """
        Embed LangChain Document objects and return them with embeddings attached.

        This is a convenience method for working with LangChain Document objects
        that includes metadata in the embedding process.
        """
        if not isinstance(documents, list):
            raise TypeError("Input must be a list of Document objects")

        if not documents:
            return []

        if not all(isinstance(doc, Document) for doc in documents):
            raise TypeError("All items must be LangChain Document objects")

        texts_to_embed = self._prepare_documents_for_embedding(documents)
        embeddings = self._get_embeddings(texts_to_embed)

        # Attach embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.metadata = doc.metadata or {}
            doc.metadata["embedding"] = embedding

        return documents

    async def aembed_langchain_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """Async version of embed_langchain_documents."""
        if not isinstance(documents, list):
            raise TypeError("Input must be a list of Document objects")

        if not documents:
            return []

        if not all(isinstance(doc, Document) for doc in documents):
            raise TypeError("All items must be LangChain Document objects")

        texts_to_embed = self._prepare_documents_for_embedding(documents)
        embeddings = await self._aget_embeddings(texts_to_embed)

        # Attach embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.metadata = doc.metadata or {}
            doc.metadata["embedding"] = embedding

        return documents
