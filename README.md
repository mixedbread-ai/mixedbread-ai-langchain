# Mixedbread AI LangChain Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-langchain.svg)](https://badge.fury.io/py/mixedbread-ai-langchain)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-langchain.svg)](https://pypi.org/project/mixedbread-ai-langchain/)

**Mixedbread AI** integration for **LangChain**, providing state-of-the-art embedding, reranking, document parsing, and retrieval.

## Overview

[Mixedbread AI](https://www.mixedbread.com) provides best-in-class embedding and reranking models, both open-source and proprietary. This integration brings comprehensive capabilities to your LangChain pipelines:

- **MixedbreadEmbeddings** - Text embeddings with full async support
- **MixedbreadReranker** - Document reranking for improved relevance  
- **MixedbreadDocumentLoader** - Advanced document loading and parsing with chunking strategies
- **MixedbreadVectorStoreRetriever** - Vector store search and retrieval
- **MixedbreadVectorStoreFileRetriever** - File-based vector store retrieval  
- **MixedbreadVectorStoreManager** - Vector store management

More information can be found in the [official documentation](https://www.mixedbread.com/docs).

## Installation

```bash
pip install mixedbread-ai-langchain
```

## Quick Start

### 1. Get your API key
Sign up at [mixedbread.com](https://www.mixedbread.com) and get your API key from the [dashboard](https://www.platform.mixedbread.com/).

### 2. Store your API key in an environment variable
```bash
export MXBAI_API_KEY="your-api-key-here"
```

### 3. Basic usage

#### Text Embeddings
```python
from mixedbread_ai_langchain import MixedbreadEmbeddings

embeddings = MixedbreadEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")

# Embed a single query
query_embedding = embeddings.embed_query("What is the capital of France?")
print(f"Query embedding dimension: {len(query_embedding)}")

# Embed multiple documents
doc_embeddings = embeddings.embed_documents([
    "Paris is the capital of France.",
    "London is the capital of England.",
    "Berlin is the capital of Germany."
])
print(f"Document embeddings: {len(doc_embeddings)} documents embedded")
```

#### Document Reranking
```python
from mixedbread_ai_langchain import MixedbreadReranker
from langchain_core.documents import Document

reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-large-v2", top_k=3)

documents = [
    Document(page_content="Paris is the capital of France."),
    Document(page_content="The Eiffel Tower is in Paris."),
    Document(page_content="London is the capital of England."),
    Document(page_content="Big Ben is in London.")
]

reranked_docs = reranker.compress_documents(
    documents=documents,
    query="What is the capital of France?"
)
print(f"Reranked {len(reranked_docs)} documents")
```

#### Document Loading and Parsing
```python
from mixedbread_ai_langchain import MixedbreadDocumentLoader

# Load a single file
loader = MixedbreadDocumentLoader(
    "path/to/document.pdf",
    chunking_strategy="page",
    return_format="markdown"
)
documents = loader.load()
print(f"Loaded {len(documents)} chunks from document")

# Load multiple files asynchronously  
import asyncio

async def load_multiple():
    loader = MixedbreadDocumentLoader([
        "path/to/doc1.pdf",
        "path/to/doc2.docx"
    ])
    documents = await loader.aload()
    return documents

documents = asyncio.run(load_multiple())

# Convenience methods for single file operations
loader = MixedbreadDocumentLoader([])  # Empty list for dynamic loading
documents = loader.load_single_file("path/to/document.pdf")
```

#### Vector Store Retrieval

##### Chunk-Level Search
```python
from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever

# Search for specific document chunks
chunk_retriever = MixedbreadVectorStoreRetriever(
    vector_store_ids=["your-vector-store-id"],
    top_k=5,
    score_threshold=0.7
)

relevant_chunks = chunk_retriever.get_relevant_documents("machine learning algorithms")
print(f"Retrieved {len(relevant_chunks)} relevant chunks")

for chunk in relevant_chunks:
    print(f"Score: {chunk.metadata['relevance_score']:.3f}")
    print(f"Source: {chunk.metadata['filename']}")
    print(f"Content: {chunk.page_content[:100]}...")
```

##### File-Level Search
```python
from mixedbread_ai_langchain import MixedbreadVectorStoreFileRetriever

# Search for entire files with optional chunk content
file_retriever = MixedbreadVectorStoreFileRetriever(
    vector_store_ids=["your-vector-store-id"],
    top_k=3,
    include_chunks=True,
    chunk_limit=2  # Include top 2 chunks per file
)

relevant_files = file_retriever.get_relevant_documents("deep learning research")
print(f"Retrieved {len(relevant_files)} relevant files")

for file_doc in relevant_files:
    print(f"File: {file_doc.metadata['filename']}")
    print(f"Score: {file_doc.metadata['relevance_score']:.3f}")
    print(f"Chunks included: {file_doc.metadata['chunks_included']}")
```

##### Vector Store Management
```python
from mixedbread_ai_langchain import MixedbreadVectorStoreManager

# Create and manage vector stores
manager = MixedbreadVectorStoreManager()

# Create a new vector store
store_id = manager.create_vector_store(
    name="Research Papers",
    description="AI/ML research paper collection",
    metadata={"project": "research", "category": "papers"}
)

# Upload files to the vector store
file_ids = manager.upload_files(store_id, ["path/to/paper1.pdf", "path/to/paper2.pdf"])
print(f"Uploaded {len(file_ids)} files")

# List files in the vector store
files = manager.list_vector_store_files(store_id)
print(f"Vector store contains {len(files)} files")

# Create retrievers for the new store
chunk_retriever = manager.create_retriever(
    vector_store_ids=[store_id],
    retriever_type="chunk",
    top_k=10
)

file_retriever = manager.create_retriever(
    vector_store_ids=[store_id], 
    retriever_type="file",
    include_chunks=True
)

# Clean up when done
manager.delete_vector_store(store_id)
```

## Components

### MixedbreadEmbeddings
- **Purpose**: Generate embeddings for text and documents
- **Features**: Async support, multiple encoding formats, customizable dimensions
- **Input**: Strings or lists of strings
- **Output**: Embedding vectors
- **Use case**: Semantic search, similarity matching, RAG pipelines

### MixedbreadReranker  
- **Purpose**: Rerank documents by relevance to a query
- **Features**: Async support, configurable top-k, metadata preservation
- **Input**: Query + List of Documents
- **Output**: Reranked documents with relevance scores
- **Use case**: Improving retrieval precision in RAG pipelines

### MixedbreadDocumentLoader
- **Purpose**: Load and parse documents from file paths (following LangChain conventions)
- **Features**: Multiple file formats, async processing, chunking strategies, element filtering
- **Input**: File paths (PDF, DOCX, PPTX, images, etc.)
- **Output**: LangChain Documents with parsed content and rich metadata
- **Use case**: Document ingestion and preprocessing for RAG pipelines
- **Methods**: Standard LangChain loader methods (`load()`, `lazy_load()`) plus convenience methods

### MixedbreadVectorStoreRetriever
- **Purpose**: Search document chunks across Mixedbread AI vector stores
- **Features**: Multi-store search, score thresholds, async support, enhanced API integration
- **Input**: Query strings
- **Output**: Relevant document chunks with metadata
- **Use case**: Semantic search and chunk-level retrieval for precise information extraction

### MixedbreadVectorStoreFileRetriever
- **Purpose**: Search entire files across Mixedbread AI vector stores
- **Features**: File-level search, optional chunk inclusion, configurable chunk limits, async support
- **Input**: Query strings
- **Output**: Relevant files with optional chunk content and comprehensive metadata
- **Use case**: File discovery and document-level retrieval for broader context

### MixedbreadVectorStoreManager
- **Purpose**: Comprehensive management of Mixedbread AI vector stores and files
- **Features**: Full CRUD operations, file management, retriever factory, async support
- **Input**: Various parameters for store/file operations
- **Output**: Vector store IDs, file IDs, metadata, and configured retrievers
- **Use case**: Vector store lifecycle management and setup

## Advanced Usage

### Async Operations
All components support async operations with direct client access for optimal performance:

```python
import asyncio
from mixedbread_ai_langchain import (
    MixedbreadEmbeddings, 
    MixedbreadReranker,
    MixedbreadVectorStoreRetriever,
    MixedbreadVectorStoreManager
)

async def async_pipeline():
    # Initialize components
    embeddings = MixedbreadEmbeddings()
    reranker = MixedbreadReranker()
    manager = MixedbreadVectorStoreManager()
    
    # Async vector store management (direct client access)
    store_response = await manager.aclient.vector_stores.create(
        name="Async Research Store",
        description="Created asynchronously"
    )
    store_id = store_response.id
    
    # Async embedding
    query_embedding = await embeddings.aembed_query("machine learning")
    
    # Async reranking
    reranked_docs = await reranker.acompress_documents(documents, "AI research")
    
    # Async vector store search (multiple options)
    retriever = MixedbreadVectorStoreRetriever(vector_store_ids=[store_id])
    
    # Option 1: Using convenience method
    search_response = await retriever.search_async("neural networks")
    
    # Option 2: Using direct client access
    direct_response = await retriever.aclient.vector_stores.search(
        query="neural networks",
        vector_store_ids=[store_id],
        top_k=5
    )
    
    # Cleanup (direct client access)
    await manager.aclient.vector_stores.delete(store_id)
    
    return query_embedding, reranked_docs, search_response

results = asyncio.run(async_pipeline())
```

### Direct Client Access

For maximum flexibility and performance, you can access the underlying Mixedbread clients directly:

```python
from mixedbread_ai_langchain import MixedbreadVectorStoreManager

# Initialize manager with both sync and async clients
manager = MixedbreadVectorStoreManager()

# Sync operations using convenience methods
store_id = manager.create_vector_store("My Store")

# Sync operations using direct client access
store_info = manager.client.vector_stores.retrieve(store_id)
files_response = manager.client.vector_stores.files.list(store_id)

# Async operations using direct client access
async def async_operations():
    # Create store
    store_response = await manager.aclient.vector_stores.create(name="Async Store")
    
    # Upload file
    with open("document.pdf", "rb") as f:
        file_response = await manager.aclient.vector_stores.files.upload_and_poll(
            vector_store_id=store_response.id, file=f
        )
    
    # Search
    search_response = await manager.aclient.vector_stores.search(
        query="machine learning",
        vector_store_ids=[store_response.id],
        top_k=5
    )
    
    return search_response

# Benefits of direct client access:
# - Full access to all SDK features
# - No wrapper overhead
# - Automatic updates when SDK is updated
# - Consistent with Mixedbread SDK patterns
```

### Custom Models
```python
# Use different models
embedder = MixedbreadEmbeddings(model="mixedbread-ai/mxbai-embed-2d-large-v1")
reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-xsmall-v1")
```

### Error Handling and Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

embeddings = MixedbreadEmbeddings()
try:
    result = embeddings.embed_query("test query")
except Exception as e:
    print(f"Embedding failed: {e}")
```

### Serialization Support
```python
# Serialize components for persistence
embeddings = MixedbreadEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
embeddings_dict = embeddings.to_dict()

# Restore from dict
restored_embeddings = MixedbreadEmbeddings.from_dict(embeddings_dict)
```

## Examples

Complete examples are available in the [`examples/`](./examples/) directory:

- **[Text Embeddings](./examples/mixedbread_embeddings.py)** - Basic and advanced embedding usage
- **[Document Reranking](./examples/mixedbread_reranking.py)** - Reranking for improved relevance
- **[Document Loading](./examples/mixedbread_document_loading.py)** - File loading and parsing
- **[Vector Store Management](./examples/mixedbread_vector_store_management.py)** - Complete vector store CRUD operations
- **[Chunk Retrieval](./examples/mixedbread_vector_store_chunks.py)** - Document chunk search and retrieval
- **[File Retrieval](./examples/mixedbread_vector_store_files.py)** - File-level search with chunk inclusion

## Testing

The package includes comprehensive tests:

```bash
# Run all tests
python run_tests.py all

# Run only unit tests (no API key needed)
python run_tests.py unit

# Run only integration tests (requires API key)
python run_tests.py integration

# Run specific component tests
python run_tests.py embeddings
python run_tests.py reranker
python run_tests.py document_loader
python run_tests.py retrievers
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

