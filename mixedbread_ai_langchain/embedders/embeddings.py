from typing import List, Optional, Union, Dict, Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import Secret
from pydantic import BaseModel, Field

from mixedbread.types import EmbeddingCreateResponse
from .types import MixedbreadEmbeddingType, EmbeddingMetadata
from ..common.client import MixedbreadClient
from ..common.mixins import SerializationMixin, AsyncMixin, ErrorHandlingMixin
from ..common.utils import create_response_meta, create_empty_embedding_response
from ..common.logging import get_logger

logger = get_logger(__name__)


class MixedbreadEmbeddings(Embeddings, BaseModel, SerializationMixin, AsyncMixin, ErrorHandlingMixin):
    """
    Mixedbread AI embeddings integration for LangChain.

    This class provides text embedding capabilities using the Mixedbread AI API,
    integrating with LangChain's standard Embeddings interface with enhanced 
    async support, error handling, and metadata tracking.
    """
    
    model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="The Mixedbread model to use for embeddings"
    )
    normalized: bool = Field(
        default=True,
        description="Whether to normalize the embeddings"
    )
    encoding_format: MixedbreadEmbeddingType = Field(
        default=MixedbreadEmbeddingType.FLOAT,
        description="Format for the embeddings"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Target dimensions for the embeddings"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt to use for embeddings"
    )
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        api_key: Union[Secret, str, None] = None,
        normalized: bool = True,
        encoding_format: Union[
            str, MixedbreadEmbeddingType
        ] = MixedbreadEmbeddingType.FLOAT,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
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
            **kwargs: Additional arguments passed to parent classes.
        """
        if isinstance(encoding_format, str):
            encoding_format = MixedbreadEmbeddingType.from_str(encoding_format)

        super().__init__(
            model=model,
            normalized=normalized,
            encoding_format=encoding_format,
            dimensions=dimensions,
            prompt=prompt,
            **kwargs
        )

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
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
            
        try:
            response: EmbeddingCreateResponse = self._client.client.embed(
                model=self.model,
                input=[text],
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embedding = response.data[0].embedding if response.data else []
            
            # Log metadata for debugging
            meta = create_response_meta(response, include_embedder_fields=True)
            logger.debug(f"Query embedding completed: {meta}")
            
            return embedding
            
        except Exception as e:
            return self._handle_api_error(e, "query embedding", [])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        if not texts:
            logger.info("Empty text list provided for embedding")
            return []
            
        # Filter out empty texts but maintain positions
        non_empty_texts = []
        text_positions = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                text_positions.append(i)
        
        if not non_empty_texts:
            logger.warning("All provided texts are empty")
            return [[] for _ in texts]

        try:
            response: EmbeddingCreateResponse = self._client.client.embed(
                model=self.model,
                input=non_empty_texts,
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embeddings = [item.embedding for item in response.data] if response.data else []
            
            # Reconstruct full results with empty embeddings for empty texts
            full_results = [[] for _ in texts]
            for i, embedding in enumerate(embeddings):
                if i < len(text_positions):
                    full_results[text_positions[i]] = embedding
                    
            # Log metadata for debugging
            meta = create_response_meta(response, include_embedder_fields=True)
            logger.debug(f"Document embedding completed: {meta}")
            
            return full_results
            
        except Exception as e:
            return self._handle_api_error(e, "document embedding", [[] for _ in texts])

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async version of embed_query.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        if not text.strip():
            logger.warning("Empty text provided for async embedding")
            return []
            
        try:
            response: EmbeddingCreateResponse = await self._client.async_client.embed(
                model=self.model,
                input=[text],
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embedding = response.data[0].embedding if response.data else []
            
            # Log metadata for debugging
            meta = create_response_meta(response, include_embedder_fields=True)
            logger.debug(f"Async query embedding completed: {meta}")
            
            return embedding
            
        except Exception as e:
            return self._handle_api_error(e, "async query embedding", [])

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed_documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        if not texts:
            logger.info("Empty text list provided for async embedding")
            return []
            
        # Filter out empty texts but maintain positions
        non_empty_texts = []
        text_positions = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                text_positions.append(i)
        
        if not non_empty_texts:
            logger.warning("All provided texts are empty")
            return [[] for _ in texts]

        try:
            response: EmbeddingCreateResponse = await self._client.async_client.embed(
                model=self.model,
                input=non_empty_texts,
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embeddings = [item.embedding for item in response.data] if response.data else []
            
            # Reconstruct full results with empty embeddings for empty texts
            full_results = [[] for _ in texts]
            for i, embedding in enumerate(embeddings):
                if i < len(text_positions):
                    full_results[text_positions[i]] = embedding
                    
            # Log metadata for debugging
            meta = create_response_meta(response, include_embedder_fields=True)
            logger.debug(f"Async document embedding completed: {meta}")
            
            return full_results
            
        except Exception as e:
            return self._handle_api_error(e, "async document embedding", [[] for _ in texts])
            
    def embed_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Embed a single text and return both embedding and metadata.
        
        Args:
            text: The text to embed.
            
        Returns:
            Dictionary containing embedding and metadata.
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding with metadata")
            return create_empty_embedding_response(self.model)
            
        try:
            response: EmbeddingCreateResponse = self._client.client.embed(
                model=self.model,
                input=[text],
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embedding = response.data[0].embedding if response.data else []
            meta = create_response_meta(response, include_embedder_fields=True)

            return {"embedding": embedding, "meta": meta}
            
        except Exception as e:
            logger.error(f"Error during embedding with metadata: {str(e)}")
            raise
            
    async def aembed_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Async version of embed_with_metadata.
        
        Args:
            text: The text to embed.
            
        Returns:
            Dictionary containing embedding and metadata.
        """
        if not text.strip():
            logger.warning("Empty text provided for async embedding with metadata")
            return create_empty_embedding_response(self.model)
            
        try:
            response: EmbeddingCreateResponse = await self._client.async_client.embed(
                model=self.model,
                input=[text],
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=self.prompt,
            )

            embedding = response.data[0].embedding if response.data else []
            meta = create_response_meta(response, include_embedder_fields=True)

            return {"embedding": embedding, "meta": meta}
            
        except Exception as e:
            logger.error(f"Error during async embedding with metadata: {str(e)}")
            raise
