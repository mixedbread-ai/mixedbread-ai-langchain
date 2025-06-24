from typing import List, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import SecretStr
from mixedbread import Mixedbread, AsyncMixedbread


class MixedbreadEmbeddings(Embeddings):
    """
    Mixedbread embeddings integration for LangChain.

    Implementation that provides text embedding capabilities using the
    Mixedbread API while following LangChain's standard Embeddings interface.
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        api_key: Union[SecretStr, str, None] = None,
        base_url: Optional[str] = None,
        normalized: bool = True,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
    ):
        """
        Initialize the Mixedbread embeddings.

        Args:
            model: The Mixedbread model to use for embeddings.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            base_url: Base URL for the API.
            normalized: Whether to normalize the embeddings.
            encoding_format: Format for the embeddings.
            dimensions: Target dimensions for the embeddings.
            prompt: Optional prompt to use for embeddings.
        """
        if api_key is None:
            api_key = get_from_dict_or_env({}, "api_key", "MXBAI_API_KEY")

        if isinstance(api_key, str):
            api_key = SecretStr(api_key)

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.normalized = normalized
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.prompt = prompt

        # Initialize clients
        resolved_api_key = self.api_key.get_secret_value()
        client_kwargs = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Mixedbread(**client_kwargs)
        self._async_client = AsyncMixedbread(**client_kwargs)

    def _filter_empty_texts(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """
        Filter out empty texts but maintain positions.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            Tuple of (non_empty_texts, text_positions)
        """
        non_empty_texts = []
        text_positions = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                text_positions.append(i)
        return non_empty_texts, text_positions
    
    def _reconstruct_results(self, embeddings: List[List[float]], text_positions: List[int], total_texts: int) -> List[List[float]]:
        """
        Reconstruct full results with empty embeddings for empty texts.
        
        Args:
            embeddings: List of embeddings from API
            text_positions: Original positions of non-empty texts
            total_texts: Total number of original texts
            
        Returns:
            Full results list with empty lists for empty texts
        """
        full_results = [[] for _ in range(total_texts)]
        for i, embedding in enumerate(embeddings):
            if i < len(text_positions):
                full_results[text_positions[i]] = embedding
        return full_results

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        if not text.strip():
            return []

        response = self._client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format,
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

        non_empty_texts, text_positions = self._filter_empty_texts(texts)

        if not non_empty_texts:
            return [[] for _ in texts]

        response = self._client.embed(
            model=self.model,
            input=non_empty_texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        embeddings = [item.embedding for item in response.data] if response.data else []

        return self._reconstruct_results(embeddings, text_positions, len(texts))

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async version of embed_query.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        if not text.strip():
            return []

        response = await self._async_client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format,
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

        non_empty_texts, text_positions = self._filter_empty_texts(texts)

        if not non_empty_texts:
            return [[] for _ in texts]

        response = await self._async_client.embed(
            model=self.model,
            input=non_empty_texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

        embeddings = [item.embedding for item in response.data] if response.data else []

        return self._reconstruct_results(embeddings, text_positions, len(texts))
