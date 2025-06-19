import asyncio
from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever


def chunk_search_example():
    """Basic chunk search usage."""
    print("=== Chunk Search Example ===")

    # Initialize retriever for chunk-level search
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=["your-vector-store-id"],
        search_type="chunk",  # Default - searches individual chunks
        top_k=5,
        # api_key="your-api-key"  # or set MXBAI_API_KEY env var
    )

    # Search for relevant chunks
    query = "What is machine learning?"
    documents = retriever.invoke(query)  # or retriever.get_relevant_documents(query)

    print(f"Query: {query}")
    print(f"Found {len(documents)} relevant chunks:")
    print()

    for i, doc in enumerate(documents):
        print(f"Chunk {i+1}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Source: {doc.metadata.get('filename', 'unknown')}")
        print(f"  Score: {doc.metadata.get('score', 'N/A'):.3f}")
        print()


def file_search_example():
    """File-level search usage."""
    print("=== File Search Example ===")

    # Initialize retriever for file-level search
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=["your-vector-store-id"],
        search_type="file",  # Searches files and aggregates chunk content
        top_k=3,
    )

    # Search for relevant files
    query = "machine learning"  # Use a query that matches our data better
    documents = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Found {len(documents)} relevant files:")
    print()

    for i, doc in enumerate(documents):
        print(f"File {i+1}:")
        print(f"  Source: {doc.metadata.get('filename', 'unknown')}")
        print(f"  File ID: {doc.metadata.get('file_id', 'N/A')}")
        print(f"  Score: {doc.metadata.get('score', 'N/A'):.3f}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Preview: {doc.page_content[:150]}...")
        print()


async def async_search_example():
    """Asynchronous search usage."""
    print("=== Async Search Example ===")

    # Initialize retriever
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=[
            "your-vector-store-id",
            "your-vector-store-id",
        ],  # Multiple stores
        search_type="chunk",
        top_k=4,
    )

    # Async search
    query = "deep learning neural networks"
    documents = await retriever.ainvoke(query)

    print(f"Async Query: {query}")
    print(f"Searched across {len(retriever.vector_store_identifiers)} vector stores")
    print(f"Found {len(documents)} results:")

    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc.page_content[:80]}...")
        print(f"     Score: {doc.metadata.get('score', 'N/A'):.3f}")
    print()


def multiple_stores_example():
    """Search across multiple vector stores."""
    print("=== Multiple Vector Stores Example ===")

    # Search across multiple vector stores
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=[
            "your-vector-store-id",
            "your-vector-store-id",
        ],
        search_type="chunk",
        top_k=6,  # Get 6 results total across all stores
    )

    query = "transformer architecture attention mechanism"
    documents = retriever.invoke(query)

    print(f"Query: {query}")
    print(f"Searched across {len(retriever.vector_store_identifiers)} vector stores")
    print(f"Results: {len(documents)} chunks")
    print()

    # Group results by source
    sources = {}
    for doc in documents:
        source = doc.metadata.get("filename", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(doc)

    for source, docs in sources.items():
        print(f"From {source}: {len(docs)} results")
        for doc in docs:
            print(
                f"  - Score: {doc.metadata.get('score', 0):.3f} | {doc.page_content[:60]}..."
            )
    print()


def main():
    """Run all examples."""
    # Note: Set MXBAI_API_KEY environment variable and update vector store IDs before running

    print("Mixedbread Vector Store Retriever Examples")
    print("=" * 50)
    print()

    try:
        # Sync examples
        chunk_search_example()
        file_search_example()
        multiple_stores_example()

        # Async example
        print("Running async example...")
        asyncio.run(async_search_example())

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Setup checklist:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Update vector_store_identifiers with your actual store IDs")
        print("3. Ensure your vector stores contain indexed documents")


if __name__ == "__main__":
    main()
