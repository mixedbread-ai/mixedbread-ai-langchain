from enum import Enum
from typing import Dict, List, Any, Union, TypedDict, Optional


class MixedbreadEmbeddingType(Enum):
    """
    Supported encoding formats for Mixedbread Embeddings API.
    """

    FLOAT = "float"
    FLOAT16 = "float16"
    BASE64 = "base64"
    BINARY = "binary"
    UBINARY = "ubinary"
    INT8 = "int8"
    UINT8 = "uint8"

    def __str__(self) -> str:
        """Return the string value of the embedding type."""
        return self.value

    @staticmethod
    def from_str(s: str) -> "MixedbreadEmbeddingType":
        """
        Convert a string to a MixedbreadEmbeddingType enum.

        Args:
            s: String representation of the embedding type.

        Returns:
            Corresponding MixedbreadEmbeddingType enum value.

        Raises:
            ValueError: If the string is not a valid embedding type.
        """
        try:
            return MixedbreadEmbeddingType(s.lower())
        except ValueError as e:
            raise ValueError(
                f"Unknown Mixedbread embedding type '{s}'. Supported types are: {[e.value for e in MixedbreadEmbeddingType]}"
            ) from e


class EmbeddingMetadata(TypedDict, total=False):
    """
    Metadata structure for embedding responses.
    
    Attributes:
        model: Name of the embedding model used.
        usage: Token usage statistics.
        normalized: Whether embeddings are normalized.
        encoding_format: Format of the returned embeddings.
        dimensions: Dimensionality of the embeddings.
        object: API response object type.
    """
    model: str
    usage: Dict[str, int]
    normalized: bool
    encoding_format: Union[str, List[str]]
    dimensions: Optional[int]
    object: Optional[str]


class RerankingMetadata(TypedDict, total=False):
    """
    Metadata structure for reranking responses.
    
    Attributes:
        model: Name of the reranking model used.
        usage: Token usage statistics.
        top_k: Number of results returned.
        object: API response object type.
    """
    model: str
    usage: Dict[str, int]
    top_k: int
    object: Optional[str]


class DocumentParsingMetadata(TypedDict, total=False):
    """
    Metadata structure for document parsing responses.
    
    Attributes:
        file_path: Original file path/name.
        parsing_job_id: ID of the parsing job.
        chunking_strategy: Strategy used for chunking.
        return_format: Format of returned content.
        element_types: Types of elements extracted.
        total_chunks: Total number of chunks created.
        chunk_index: Index of this specific chunk.
        elements: List of elements in this chunk.
        pages: Page numbers this chunk spans.
    """
    file_path: str
    parsing_job_id: Optional[str]
    chunking_strategy: Optional[str]
    return_format: Optional[str]
    element_types: Optional[List[str]]
    total_chunks: int
    chunk_index: int
    elements: List[Dict[str, Any]]
    pages: Optional[List[int]]
    page_range: Optional[str]
