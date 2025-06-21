import asyncio
from mixedbread_ai_langchain import MixedbreadEmbeddings


def basic_usage():
    """Basic synchronous embedding usage."""
    print("=== Basic Embedding Usage ===")

    # Initialize embeddings
    embeddings = MixedbreadEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
        # api_key="your-api-key"  # or set MXBAI_API_KEY env var
    )

    # Embed a single query
    query = "What is machine learning?"
    query_embedding = embeddings.embed_query(query)
    print(f"Query: {query}")
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")
    print()

    # Embed multiple documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
    ]

    doc_embeddings = embeddings.embed_documents(documents)
    print(f"Embedded {len(doc_embeddings)} documents")
    for i, doc in enumerate(documents):
        print(f"Doc {i+1}: {doc[:50]}...")
        print(f"  Embedding dimension: {len(doc_embeddings[i])}")
    print()


async def async_usage():
    """Asynchronous embedding usage."""
    print("=== Async Embedding Usage ===")

    # Initialize embeddings
    embeddings = MixedbreadEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
    )

    # Async embed query
    query = "What is deep learning?"
    query_embedding = await embeddings.aembed_query(query)
    print(f"Async query embedding dimension: {len(query_embedding)}")

    # Async embed documents
    documents = [
        "Transformers revolutionized natural language processing.",
        "BERT is a bidirectional encoder representation model.",
    ]

    doc_embeddings = await embeddings.aembed_documents(documents)
    print(f"Async embedded {len(doc_embeddings)} documents")
    print()


def custom_configuration():
    """Example with custom configuration."""
    print("=== Custom Configuration ===")

    # Initialize with custom settings
    embeddings = MixedbreadEmbeddings(
        model="mixedbread-ai/mxbai-embed-2d-large-v1",  # Different model
        normalized=False,  # Don't normalize
        encoding_format="float16",  # Use float16
        dimensions=512,  # Custom dimensions
        prompt="Represent this sentence for semantic search:",  # Custom prompt
    )

    # Use with custom settings
    text = "Artificial intelligence is transforming industries."
    embedding = embeddings.embed_query(text)
    print(f"Custom config embedding dimension: {len(embedding)}")
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
