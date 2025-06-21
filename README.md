# Mixedbread AI LangChain Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-langchain.svg)](https://badge.fury.io/py/mixedbread-ai-langchain)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-langchain.svg)](https://pypi.org/project/mixedbread-ai-langchain/)

**Mixedbread AI** integration for **LangChain**, providing state-of-the-art embedding, reranking, document parsing, and retrieval capabilities.

## Components

- **MixedbreadEmbeddings** - Text embeddings with async support
- **MixedbreadReranker** - Document reranking for improved relevance  
- **MixedbreadDocumentLoader** - Multi-format document parsing and loading
- **MixedbreadVectorStoreRetriever** - Vector store search and retrieval

## Installation

```bash
pip install mixedbread-ai-langchain
```

## Quick Start

Get your API key from the [Mixedbread Platform](https://www.platform.mixedbread.com/) and set it as an environment variable:

```bash
export MXBAI_API_KEY="your-api-key"
```

### Basic Usage

```python
from mixedbread_ai_langchain import MixedbreadEmbeddings

# Embed text
embeddings = MixedbreadEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
result = embeddings.embed_query("What is the capital of France?")
```

## Async Support

All components support async operations:

```python
import asyncio

async def embed_text():
    embeddings = MixedbreadEmbeddings()
    result = await embeddings.aembed_query("Async embedding example")
    return result

# Run async
embedding = asyncio.run(embed_text())
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples:

- **[Embeddings](./examples/embeddings_example.py)** - Text and document embedding
- **[Reranker](./examples/reranker_example.py)** - Document reranking  
- **[Document Loader](./examples/document_loader_example.py)** - File parsing and loading
- **[Vector Retriever](./examples/retriever_example.py)** - Vector-based search

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ 

# Run specific test files
pytest tests/test_embeddings.py
```

## Documentation

Learn more at [mixedbread.com/docs](https://www.mixedbread.com/docs):
- [Embeddings API](https://www.mixedbread.com/docs/embeddings/overview)
- [Reranking API](https://www.mixedbread.com/docs/reranking/overview)  
- [Parsing API](https://www.mixedbread.com/docs/parsing/overview)
- [Vector Stores API](https://www.mixedbread.com/docs/vector-stores/overview)

## License

Apache 2.0 License

