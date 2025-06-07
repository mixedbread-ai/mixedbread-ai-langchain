"""Shared utilities for mixedbread-ai-langchain components."""

from typing import Dict, List, Any, Optional, Union, Sequence
from langchain_core.documents import Document


def validate_documents(documents: Union[List[Document], Sequence[Document]]) -> None:
    """
    Validate that input is a list/sequence of LangChain Documents.
    
    Args:
        documents: List/sequence of documents to validate.
        
    Raises:
        TypeError: If input is not a list/sequence of LangChain Documents.
    """
    if not isinstance(documents, (list, tuple)) or (
        documents and not isinstance(documents[0], Document)
    ):
        raise TypeError("Input must be a list or sequence of LangChain Documents.")


def create_response_meta(
    response: Any,
    include_embedder_fields: bool = False,
    include_reranker_fields: bool = False
) -> Dict[str, Any]:
    """
    Create standardized response metadata from API response.
    
    Args:
        response: API response object with model, usage, etc.
        include_embedder_fields: Include embedder-specific fields (normalized, encoding_format, dimensions).
        include_reranker_fields: Include reranker-specific fields if any.
        
    Returns:
        Standardized metadata dictionary.
    """
    meta = {
        "model": getattr(response, "model", "unknown"),
        "usage": getattr(response, "usage", {}).model_dump() if hasattr(getattr(response, "usage", {}), "model_dump") else getattr(response, "usage", {}),
        "object": getattr(response, "object", "unknown"),
    }
    
    if include_embedder_fields:
        meta.update({
            "normalized": getattr(response, "normalized", True),
            "encoding_format": getattr(response, "encoding_format", "float"),
            "dimensions": getattr(response, "dimensions", None),
        })
    
    return meta


def create_empty_response(
    response_type: str,
    model: str = "unknown"
) -> Dict[str, Any]:
    """
    Create standardized empty response for different component types.
    
    Args:
        response_type: Type of response ("embedding", "documents", "reranking").
        model: Model name to include in metadata.
        
    Returns:
        Empty response with appropriate structure.
    """
    empty_usage = {
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
    
    base_meta = {
        "model": model,
        "usage": empty_usage,
        "object": "list",
    }
    
    if response_type == "embedding":
        return {
            "embedding": [],
            "meta": {
                **base_meta,
                "normalized": True,
                "encoding_format": "float",
                "dimensions": 0,
            }
        }
    elif response_type == "documents":
        return {
            "documents": [],
            "meta": {
                **base_meta,
                "normalized": True,
                "encoding_format": "float", 
                "dimensions": 0,
            }
        }
    elif response_type == "reranking":
        return {
            "documents": [],
            "meta": {
                **base_meta,
                "top_k": 0,
            }
        }
    else:
        raise ValueError(f"Unknown response type: {response_type}")


def create_empty_documents_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for document-based operations."""
    return create_empty_response("documents", model)


def create_empty_embedding_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for embedding operations."""
    return create_empty_response("embedding", model)


def create_empty_reranking_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for reranking operations."""
    return create_empty_response("reranking", model)


def prepare_documents_for_processing(documents: Sequence[Document]) -> List[str]:
    """
    Extract text content from documents for API processing.
    
    Args:
        documents: Sequence of LangChain documents.
        
    Returns:
        List of text strings to process.
    """
    return [doc.page_content or "" for doc in documents]


def create_error_document(
    error_msg: str, 
    source: str = "unknown",
    meta: Optional[Dict[str, Any]] = None
) -> Document:
    """
    Create an error document when processing fails.
    
    Args:
        error_msg: Error message to include.
        source: Source identifier for the error.
        meta: Additional metadata.
        
    Returns:
        Error document with appropriate metadata.
    """
    error_metadata = {
        "source": source,
        "processing_error": error_msg,
        "processing_status": "failed",
        **(meta or {}),
    }
    
    return Document(
        page_content="",
        metadata=error_metadata
    )