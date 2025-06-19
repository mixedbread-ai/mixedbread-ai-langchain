import asyncio
from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadReranker


def basic_usage():
    """Basic synchronous reranking usage."""
    print("=== Basic Reranking Usage ===")

    # Initialize reranker
    reranker = MixedbreadReranker(
        model="mixedbread-ai/mxbai-rerank-large-v2",
        top_k=3,
        # api_key="your-api-key"  # or set MXBAI_API_KEY env var
    )

    # Create some documents to rerank
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"source": "ml_guide.pdf", "page": 1},
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers for complex pattern recognition.",
            metadata={"source": "dl_tutorial.pdf", "page": 5},
        ),
        Document(
            page_content="Natural language processing helps computers understand and generate human language.",
            metadata={"source": "nlp_book.pdf", "page": 12},
        ),
        Document(
            page_content="Computer vision enables machines to interpret and analyze visual information.",
            metadata={"source": "cv_paper.pdf", "page": 3},
        ),
        Document(
            page_content="Reinforcement learning trains agents through trial and error with rewards.",
            metadata={"source": "rl_research.pdf", "page": 8},
        ),
    ]

    # Rerank documents based on a query
    query = "What is deep learning and neural networks?"
    reranked_docs = reranker.compress_documents(documents, query)

    print(f"Query: {query}")
    print(f"Original documents: {len(documents)}")
    print(f"Reranked documents: {len(reranked_docs)}")
    print()

    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}:")
        print(f"  Content: {doc.page_content[:80]}...")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Rerank Score: {doc.metadata.get('rerank_score', 'N/A'):.3f}")
        print(f"  Original Index: {doc.metadata.get('rerank_index', 'N/A')}")
        print()


async def async_usage():
    """Asynchronous reranking usage."""
    print("=== Async Reranking Usage ===")

    # Initialize reranker
    reranker = MixedbreadReranker(
        model="mixedbread-ai/mxbai-rerank-large-v2",
        top_k=2,
    )

    # Create documents about AI topics
    documents = [
        Document(
            page_content="Transformers revolutionized natural language processing with attention mechanisms."
        ),
        Document(
            page_content="BERT is a bidirectional encoder representation model for language understanding."
        ),
        Document(
            page_content="GPT models use autoregressive generation for text completion tasks."
        ),
        Document(
            page_content="Vision Transformers apply transformer architecture to image classification."
        ),
    ]

    # Async reranking
    query = "What are transformer models in NLP?"
    reranked_docs = await reranker.acompress_documents(documents, query)

    print(f"Async Query: {query}")
    print(f"Top {len(reranked_docs)} results:")

    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. {doc.page_content}")
        print(f"     Score: {doc.metadata.get('rerank_score', 'N/A'):.3f}")
    print()


def custom_configuration():
    """Example with custom configuration."""
    print("=== Custom Configuration ===")

    # Initialize with custom settings
    reranker = MixedbreadReranker(
        model="mixedbread-ai/mxbai-rerank-xsmall-v1",  # Different model
        top_k=2,  # Only top 2 results
        return_input=False,  # Don't return input text in API response
    )

    # Create documents
    documents = [
        Document(page_content="Python is a high-level programming language."),
        Document(page_content="JavaScript is commonly used for web development."),
        Document(
            page_content="Rust provides memory safety without garbage collection."
        ),
        Document(page_content="Go is designed for scalable network services."),
    ]

    # Rerank with custom configuration
    query = "Which language is best for web development?"
    reranked_docs = reranker.compress_documents(documents, query)

    print(f"Custom config query: {query}")
    print(f"Results with top_k={reranker.top_k}:")

    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. {doc.page_content}")
        print(f"     Score: {doc.metadata.get('rerank_score', 'N/A'):.3f}")
    print()


def main():
    """Run all examples."""
    # Note: Set MXBAI_API_KEY environment variable before running

    try:
        basic_usage()
        custom_configuration()
        # Run async example
        asyncio.run(async_usage())

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set MXBAI_API_KEY environment variable")


if __name__ == "__main__":
    main()
