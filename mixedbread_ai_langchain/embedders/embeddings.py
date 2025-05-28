from typing import List, Optional, Union

from langchain_core.embeddings import Embeddings

from mixedbread.types import EmbeddingCreateResponse
from .types import MixedbreadEmbeddingType
from ..common.client import MixedbreadClient


class MixedbreadEmbeddings(Embeddings):
    """
    Mixedbread AI embeddings integration for LangChain.

    This class provides text embedding capabilities using the Mixedbread AI API,
    integrating with LangChain's standard Embeddings interface.
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        api_key: Optional[str] = None,
        normalized: bool = True,
        encoding_format: Union[
            str, MixedbreadEmbeddingType
        ] = MixedbreadEmbeddingType.FLOAT,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the Mixedbread embeddings.

        Args:
            model: The Mixedbread model to use for embeddings.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            normalized: Whether to normalize the embeddings.
            encoding_format: Format for the embeddings (float, float16, base64, binary, ubinary, int8, uint8).
            dimensions: Target dimensions for the embeddings.
            prompt: Optional prompt to use for embeddings.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.model = model
        self.normalized = normalized

        if isinstance(encoding_format, str):
            self.encoding_format = MixedbreadEmbeddingType.from_str(encoding_format)
        else:
            self.encoding_format = encoding_format

        self.dimensions = dimensions
        self.prompt = prompt

        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        response: EmbeddingCreateResponse = self._client.client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        return response.data[0].embedding if response.data else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        if not texts:
            return []

        response: EmbeddingCreateResponse = self._client.client.embed(
            model=self.model,
            input=texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        return [item.embedding for item in response.data] if response.data else []

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async version of embed_query.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        response: EmbeddingCreateResponse = await self._client.async_client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        return response.data[0].embedding if response.data else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed_documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        if not texts:
            return []

        response: EmbeddingCreateResponse = await self._client.async_client.embed(
            model=self.model,
            input=texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        return [item.embedding for item in response.data] if response.data else []
