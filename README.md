# Mixedbread AI Langchain Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-langchain.svg)](https://badge.fury.io/py/mixedbread-ai-langchain)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-langchain.svg)](https://pypi.org/project/mixedbread-ai-langchain/)

**Mixedbread AI** integration for **LangChain**. This package provides seamless access to Mixedbread's multimodal AI capabilities, enabling intelligent search that understands meaning across text, images, code, PDFs, and diverse document types. Use our state of the art embedding and reranking models as part of your langchain workflows.

## Components

- **MixedbreadEmbeddings** - State-of-the-art embedding models that generate vectors capturing deep contextual meaning, with full async support and batch processing capabilities for transforming unstructured data into intelligent search
- **MixedbreadReranker** - Powerful semantic reranking that significantly boosts search relevance by applying sophisticated models to reorder initial search results, essential for optimizing RAG applications and improving precision
- **MixedbreadDocumentLoader** - Layout-aware document parsing supporting PDF, PPTX, HTML and more formats, providing structured output with detailed content elements for high-quality downstream processing
- **MixedbreadVectorStoreRetriever** - AI-native search engine that enables conversational queries across multimodal data, supporting millions of documents with natural language understanding across multiple languages.

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

embeddings = MixedbreadEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
result = embeddings.embed_query("Who is German and likes bread?")
```

## Async Support

All components support async operations:

```python
import asyncio

async def embed_text():
    embeddings = MixedbreadEmbeddings()
    result = await embeddings.aembed_query("Async embedding example")
    return result

embedding = asyncio.run(embed_text())
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples:

- **[Embeddings](https://github.com/mixedbread-ai/mixedbread-ai-langchain/blob/main/examples/embeddings_example.py)** - Text and document embedding
- **[Reranker](https://github.com/mixedbread-ai/mixedbread-ai-langchain/blob/main/examples/reranker_example.py)** - Document reranking
- **[Document Loader](https://github.com/mixedbread-ai/mixedbread-ai-langchain/blob/main/examples/document_loader_example.py)** - File parsing and loading
- **[Vector Retriever](https://github.com/mixedbread-ai/mixedbread-ai-langchain/blob/main/examples/retriever_example.py)** - Vector-based search

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
python run_tests.py all

# Run unit tests only (no API key required)
python run_tests.py unit

# Run integration tests (requires API key)
python run_tests.py integration

# Run specific test files
python run_tests.py tests/test_embeddings.py
```

## Documentation

Learn more at [mixedbread.com/docs](https://www.mixedbread.com/docs):

- [Embeddings API](https://www.mixedbread.com/docs/embeddings/overview)
- [Reranking API](https://www.mixedbread.com/docs/reranking/overview)
- [Parsing API](https://www.mixedbread.com/docs/parsing/overview)
- [Vector Stores API](https://www.mixedbread.com/docs/vector-stores/overview)

## License

Apache 2.0 License
