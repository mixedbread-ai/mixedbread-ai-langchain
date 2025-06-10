import asyncio
from mixedbread_ai_langchain import MixedbreadVectorStoreFileRetriever


def file_search_with_chunks():
    """File search with chunk content included."""
    print("=== File Search with Chunks Example ===\n")

    # Create a file retriever that includes chunk content
    retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=[
            "vector_store_abc123xyz789"
        ],  # Replace with your vector store ID
        top_k=3,
        score_threshold=0.7,
        include_chunks=True,
        chunk_limit=3,  # Include up to 3 most relevant chunks per file
        return_metadata=True,
    )

    query = "What is semantic search and how does it improve information retrieval?"
    print(f"Query: {query}")

    try:
        documents = retriever.get_relevant_documents(query)
        print(f"Found {len(documents)} relevant files:")

        for i, doc in enumerate(documents, 1):
            score = doc.metadata.get("relevance_score", 0)
            filename = doc.metadata.get("filename", "Unknown")
            file_id = doc.metadata.get("file_id", "N/A")
            chunks_included = doc.metadata.get("chunks_included", 0)
            total_chunks = doc.metadata.get("total_chunks", 0)
            status = doc.metadata.get("status", "Unknown")
            usage_bytes = doc.metadata.get("usage_bytes", 0)

            print(f"\n{i}. Score: {score:.3f}")
            print(f"   File: {filename}")
            print(f"   File ID: {file_id}")
            print(f"   Status: {status}")
            print(f"   Size: {usage_bytes} bytes")
            print(f"   Chunks: {chunks_included}/{total_chunks} included")
            print(f"   Content preview: {doc.page_content[:200]}...")

            # Show chunk details if available
            chunk_details = doc.metadata.get("chunk_details", [])
            if chunk_details:
                print(f"   Chunk details:")
                for j, chunk_info in enumerate(
                    chunk_details[:2], 1
                ):  # Show first 2 chunks
                    chunk_score = chunk_info.get("score", 0)
                    chunk_pos = chunk_info.get("position", "N/A")
                    print(f"     {j}. Position: {chunk_pos}, Score: {chunk_score:.3f}")

    except Exception as e:
        print(f"Search failed: {e}")
        print("Note: Replace 'vector_store_abc123xyz789' with a valid vector store ID")


def file_only_search():
    """File search without chunk content (metadata only)."""
    print("\n=== File-Only Search Example ===\n")

    # Create a file retriever without chunk content
    file_only_retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=[
            "vector_store_abc123xyz789"
        ],  # Replace with your vector store ID
        top_k=5,
        score_threshold=0.6,
        include_chunks=False,  # Only return file metadata, no chunk content
        return_metadata=True,
    )

    queries = [
        "artificial intelligence applications",
        "machine learning techniques",
        "natural language processing",
    ]

    for query in queries:
        print(f"Query: {query}")
        try:
            file_results = file_only_retriever.get_relevant_documents(query)
            print(f"  Found {len(file_results)} files")

            for i, doc in enumerate(file_results, 1):
                filename = doc.metadata.get("filename", "Unknown")
                score = doc.metadata.get("relevance_score", 0)
                file_id = doc.metadata.get("file_id", "N/A")
                created_at = doc.metadata.get("created_at", "N/A")

                print(f"    {i}. {filename}")
                print(
                    f"       Score: {score:.3f} | ID: {file_id} | Created: {created_at}"
                )

        except Exception as e:
            print(f"  Search failed: {e}")
        print()


def multi_store_file_search():
    """Search files across multiple vector stores."""
    print("=== Multi-Store File Search Example ===\n")

    # Search across multiple vector stores
    multi_retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=[
            "store_documents_id",  # Replace with actual vector store IDs
            "store_research_id",
            "store_technical_id",
        ],
        top_k=8,
        score_threshold=0.65,
        include_chunks=True,
        chunk_limit=1,  # Just one chunk per file for preview
        return_metadata=True,
    )

    query = "deep learning and neural network architectures"
    print(f"Query: {query}")
    print(f"Searching across {len(multi_retriever.vector_store_ids)} vector stores...")

    try:
        results = multi_retriever.get_relevant_documents(query)
        print(f"Found {len(results)} files total")

        # Group results by vector store
        store_groups = {}
        for doc in results:
            store_id = doc.metadata.get("vector_store_id", "unknown")
            if store_id not in store_groups:
                store_groups[store_id] = []
            store_groups[store_id].append(doc)

        print(f"\nResults by vector store:")
        for store_id, docs in store_groups.items():
            print(f"  {store_id}: {len(docs)} files")
            for doc in docs[:2]:  # Show first 2 files from each store
                filename = doc.metadata.get("filename", "Unknown")
                score = doc.metadata.get("relevance_score", 0)
                print(f"    - {filename} (Score: {score:.3f})")

    except Exception as e:
        print(f"Multi-store search failed: {e}")


def dynamic_configuration_example():
    """Dynamic configuration of file retriever."""
    print("=== Dynamic Configuration Example ===\n")

    retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=["config_test_store"],  # Replace with your vector store ID
        top_k=5,
        include_chunks=False,
    )

    print("Initial configuration:")
    print(f"  - Include chunks: {retriever.include_chunks}")
    print(f"  - Chunk limit: {retriever.chunk_limit}")
    print(f"  - Score threshold: {retriever.score_threshold}")

    # Enable chunk inclusion dynamically
    print("\nEnabling chunk inclusion...")
    retriever.set_chunk_inclusion(include_chunks=True, chunk_limit=2)

    print("Updated configuration:")
    print(f"  - Include chunks: {retriever.include_chunks}")
    print(f"  - Chunk limit: {retriever.chunk_limit}")

    # Update search options
    print("\nUpdating search options...")
    retriever.update_search_options(
        score_threshold=0.8, return_metadata=True, return_chunks=True, chunk_limit=3
    )

    print("Final configuration:")
    print(f"  - Score threshold: {retriever.score_threshold}")
    print(f"  - Return metadata: {retriever.return_metadata}")
    print(f"  - Include chunks: {retriever.include_chunks}")
    print(f"  - Chunk limit: {retriever.chunk_limit}")


async def async_file_search():
    """Async file search example using direct client access."""
    print("\n=== Async File Search Example ===\n")

    retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=["async_file_store"],  # Replace with your vector store ID
        top_k=4,
        include_chunks=True,
        chunk_limit=1,
    )

    queries = [
        "computer vision applications",
        "reinforcement learning algorithms",
        "transformer architectures",
    ]

    print("Running concurrent file searches using direct async client...")

    # Method 1: Using convenience method
    async def search_files_with_convenience(query):
        try:
            response = await retriever.search_async(query)
            filenames = [
                file.filename for file in response.data if hasattr(file, "filename")
            ]
            return query, len(response.data), filenames
        except Exception as e:
            return query, 0, f"Error: {e}"

    # Method 2: Using direct client access
    async def search_files_with_direct_client(query):
        try:
            response = await retriever.aclient.vector_stores.files.search(
                query=query,
                vector_store_ids=retriever.vector_store_ids,
                top_k=retriever.top_k,
                return_chunks=retriever.include_chunks,
                chunk_limit=retriever.chunk_limit,
            )
            filenames = [
                file.filename for file in response.data if hasattr(file, "filename")
            ]
            return query, len(response.data), filenames
        except Exception as e:
            return query, 0, f"Error: {e}"

    print("\n1. Using convenience method (search_async):")
    tasks = [search_files_with_convenience(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for query, count, filenames in results:
        print(f"\nQuery: {query}")
        if isinstance(filenames, str):  # Error case
            print(f"  {filenames}")
        else:
            print(f"  Found {count} files:")
            for filename in filenames[:3]:  # Show first 3 filenames
                print(f"    - {filename}")

    print("\n2. Using direct client access:")
    # Example of bulk concurrent file operations
    try:
        # Create multiple file search tasks using direct client
        direct_tasks = [search_files_with_direct_client(query) for query in queries[:2]]
        direct_results = await asyncio.gather(*direct_tasks, return_exceptions=True)

        for query, count, filenames in direct_results:
            print(f"\nQuery (direct): {query}")
            if isinstance(filenames, str):
                print(f"  {filenames}")
            else:
                print(f"  Found {count} files using direct client access")

    except Exception as e:
        print(f"  Direct client search failed: {e}")


def comparison_example():
    """Compare file search vs chunk search."""
    print("=== File vs Chunk Search Comparison ===\n")

    # File-based retriever
    file_retriever = MixedbreadVectorStoreFileRetriever(
        vector_store_ids=["comparison_store"],  # Replace with your vector store ID
        top_k=3,
        include_chunks=False,
    )

    # Import chunk retriever for comparison
    from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever

    chunk_retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["comparison_store"],  # Same store
        top_k=10,  # More chunks since they're smaller units
    )

    query = "machine learning model evaluation techniques"
    print(f"Query: {query}")

    try:
        # File search
        print("\nFile search results:")
        file_results = file_retriever.get_relevant_documents(query)
        print(f"  Found {len(file_results)} files")
        for doc in file_results:
            filename = doc.metadata.get("filename", "Unknown")
            score = doc.metadata.get("relevance_score", 0)
            print(f"    - {filename} (Score: {score:.3f})")

        # Chunk search
        print("\nChunk search results:")
        chunk_results = chunk_retriever.get_relevant_documents(query)
        print(f"  Found {len(chunk_results)} chunks")
        chunk_files = set()
        for doc in chunk_results:
            filename = doc.metadata.get("filename", "Unknown")
            chunk_files.add(filename)
        print(f"  From {len(chunk_files)} unique files:")
        for filename in list(chunk_files)[:3]:
            print(f"    - {filename}")

        print(f"\nComparison:")
        print(f"  File search: {len(file_results)} files (complete files)")
        print(
            f"  Chunk search: {len(chunk_results)} chunks from {len(chunk_files)} files"
        )

    except Exception as e:
        print(f"Comparison failed: {e}")


if __name__ == "__main__":
    # Run file search with chunks
    file_search_with_chunks()

    # Run file-only search
    file_only_search()

    # Run multi-store search
    multi_store_file_search()

    # Run dynamic configuration example
    dynamic_configuration_example()

    # Run async example
    print("Running async file search example...")
    asyncio.run(async_file_search())

    # Run comparison example
    comparison_example()

    print("\n=== All file search examples completed ===")
    print(
        "\nNote: Replace example vector store IDs with your actual vector store IDs to test with real data."
    )
