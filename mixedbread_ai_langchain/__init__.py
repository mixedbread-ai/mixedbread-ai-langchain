"""
Mixedbread AI LangChain Integration.

This package provides LangChain components for integrating with Mixedbread's
embedding, reranking, document parsing, and retrieval services.

The package includes:
- Text embeddings with full async support
- Document reranking for improved relevance
- Document parsing and loading with multiple file formats
- Vector store search and retrieval
- Direct SDK integration for all services

Examples:
    Basic embedding usage:
    >>> from mixedbread_ai_langchain import MixedbreadEmbeddings
    >>> embeddings = MixedbreadEmbeddings()
    >>> result = embeddings.embed_query("Hello world")

    Document reranking:
    >>> from mixedbread_ai_langchain import MixedbreadReranker
    >>> reranker = MixedbreadReranker()
    >>> reranked = reranker.compress_documents(documents, "query")

    Document loading:
    >>> from mixedbread_ai_langchain import MixedbreadDocumentLoader
    >>> loader = MixedbreadDocumentLoader("document.pdf")
    >>> docs = loader.load()
"""

from .embedders import MixedbreadEmbeddings
from .loaders import MixedbreadDocumentLoader
from .retrievers import MixedbreadVectorStoreRetriever
from .compressors import MixedbreadReranker

# Version info
__version__ = "1.0.0"

__all__ = [
    # Core components
    "MixedbreadEmbeddings",
    "MixedbreadReranker",
    "MixedbreadDocumentLoader",
    # Retrieval components
    "MixedbreadVectorStoreRetriever",
    # Version
    "__version__",
]
