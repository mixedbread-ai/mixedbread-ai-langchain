import asyncio
from mixedbread_ai_langchain import MixedbreadVectorStoreManager


def main():
    """Main synchronous example."""
    print("=== Vector Store Management Example ===\n")

    # Initialize the vector store manager
    manager = MixedbreadVectorStoreManager()

    # 1. Create a new vector store with metadata
    print("1. Creating vector store...")
    store_id = manager.create_vector_store(
        name="Research Documents",
        description="Collection of AI/ML research papers and technical documentation",
        metadata={
            "project": "ai-research",
            "category": "technical-docs",
            "created_by": "example-script",
        },
    )
    print(f"   Created vector store: {store_id}")

    try:
        # 2. Get vector store information
        print("\n2. Getting vector store information...")
        store_info = manager.get_vector_store_info(store_id)
        print(f"   Name: {store_info.get('name')}")
        print(f"   Description: {store_info.get('description')}")
        print(f"   Metadata: {store_info.get('metadata', {})}")

        # 3. Update vector store
        print("\n3. Updating vector store...")
        updated_info = manager.update_vector_store(
            vector_store_id=store_id,
            description="Updated: AI/ML research papers with enhanced metadata",
            metadata={
                "project": "ai-research",
                "category": "technical-docs",
                "updated_by": "example-script",
                "version": "1.1",
            },
        )
        print(f"   Updated successfully")

        # 4. Upload files to vector store (if files exist)
        print("\n4. File upload example...")
        try:
            file_paths = ["./data/acme_invoice.pdf"]  # Example file path
            file_ids = manager.upload_files(store_id, file_paths)
            print(f"   Uploaded {len(file_ids)} files:")
            for i, file_id in enumerate(file_ids, 1):
                print(f"     {i}. File ID: {file_id}")

            # 5. List files in vector store
            print("\n5. Listing files in vector store...")
            files = manager.list_vector_store_files(store_id, limit=10)
            print(f"   Found {len(files)} files in vector store:")
            for file_info in files:
                print(
                    f"     - {file_info.get('filename', 'Unknown')} (ID: {file_info.get('id')})"
                )

        except Exception as e:
            print(f"   File operations skipped: {e}")
            print("   Note: Add files to ./data/ directory to test file operations")

        # 6. Create retrievers
        print("\n6. Creating retrievers...")

        # Chunk-based retriever
        chunk_retriever = manager.create_retriever(
            vector_store_ids=[store_id],
            retriever_type="chunk",
            top_k=5,
            score_threshold=0.7,
        )
        print("   Created chunk-based retriever")

        # File-based retriever
        file_retriever = manager.create_retriever(
            vector_store_ids=[store_id],
            retriever_type="file",
            top_k=3,
            include_chunks=True,
            chunk_limit=2,
        )
        print("   Created file-based retriever")

        # 7. Example searches (will work even with empty vector store)
        print("\n7. Example searches...")
        query = "machine learning algorithms and neural networks"
        print(f"   Query: '{query}'")

        try:
            chunk_results = chunk_retriever.get_relevant_documents(query)
            print(f"   Chunk retriever found: {len(chunk_results)} chunks")

            file_results = file_retriever.get_relevant_documents(query)
            print(f"   File retriever found: {len(file_results)} files")

            if chunk_results:
                print(
                    f"   Sample chunk score: {chunk_results[0].metadata.get('relevance_score', 'N/A')}"
                )
            if file_results:
                print(
                    f"   Sample file score: {file_results[0].metadata.get('relevance_score', 'N/A')}"
                )

        except Exception as e:
            print(f"   Search results: No documents found (empty store or API error)")

        # 8. List all vector stores
        print("\n8. Listing all vector stores...")
        stores = manager.list_vector_stores(limit=20)
        print(f"   Found {len(stores)} total vector stores")

        # Find our test store
        test_stores = [s for s in stores if s.get("id") == store_id]
        if test_stores:
            print(f"   Our test store is listed: {test_stores[0].get('name')}")

    finally:
        # 9. Cleanup - delete the test vector store
        print(f"\n9. Cleaning up...")
        try:
            result = manager.delete_vector_store(store_id)
            if result:
                print(f"   Successfully deleted vector store: {store_id}")
            else:
                print(f"   Failed to delete vector store: {store_id}")
        except Exception as e:
            print(f"   Cleanup error: {e}")

    print("\n=== Example completed ===")


async def async_example():
    """Async version using direct client access."""
    print("\n=== Async Vector Store Management Example ===\n")

    manager = MixedbreadVectorStoreManager()

    # Create vector store asynchronously using direct client access
    print("1. Creating vector store (async)...")
    response = await manager.aclient.vector_stores.create(
        name="Async Research Store",
        description="Async created vector store for testing",
        metadata={"async": True, "test": True},
    )
    store_id = response.id
    print(f"   Created async vector store: {store_id}")

    try:
        # Get info asynchronously using direct client
        print("\n2. Getting store info (async)...")
        store_info = await manager.aclient.vector_stores.retrieve(store_id)
        print(f"   Name: {store_info.name}")

        # Update asynchronously using direct client
        print("\n3. Updating store (async)...")
        await manager.aclient.vector_stores.update(
            vector_store_id=store_id, description="Updated via async operation"
        )
        print("   Updated successfully")

        # List stores asynchronously using direct client
        print("\n4. Listing stores (async)...")
        stores_response = await manager.aclient.vector_stores.list(limit=10)
        print(f"   Found {len(stores_response.data)} stores")

        # Example of async file upload (if files exist)
        print("\n5. Async file operations example...")
        try:
            # Upload a file asynchronously
            with open("./data/acme_invoice.pdf", "rb") as f:
                file_response = (
                    await manager.aclient.vector_stores.files.upload_and_poll(
                        vector_store_id=store_id, file=f
                    )
                )
            print(f"   Uploaded file: {file_response.id}")

            # List files asynchronously
            files_response = await manager.aclient.vector_stores.files.list(
                vector_store_id=store_id
            )
            print(f"   Files in store: {len(files_response.data)}")

        except Exception as e:
            print(f"   File operations skipped: {e}")

    finally:
        # Cleanup asynchronously using direct client
        print("\n6. Cleaning up (async)...")
        await manager.aclient.vector_stores.delete(store_id)
        print(f"   Successfully deleted async vector store: {store_id}")

    print("\n=== Async example completed ===")


def file_management_example():
    """Example focused on file management operations."""
    print("\n=== File Management Example ===\n")

    manager = MixedbreadVectorStoreManager()

    # Create a test vector store
    store_id = manager.create_vector_store(
        name="File Management Test", description="Testing file operations"
    )
    print(f"Created test vector store: {store_id}")

    try:
        # Demonstrate file operations (using example file IDs)
        print("\n1. File management operations...")

        # Note: In real usage, you would have actual file IDs from uploads
        # This demonstrates the API structure
        print("   File operations demonstrated:")
        print("   - add_file_to_vector_store(store_id, file_id)")
        print("   - list_vector_store_files(store_id)")
        print("   - get_vector_store_file_info(store_id, file_id)")
        print("   - delete_vector_store_file(store_id, file_id)")

        # List files (will be empty for new store)
        files = manager.list_vector_store_files(store_id)
        print(f"   Current files in store: {len(files)}")

    finally:
        # Cleanup
        manager.delete_vector_store(store_id)
        print(f"   Cleaned up test vector store")


if __name__ == "__main__":
    # Run the main synchronous example
    main()

    # Run the async example
    asyncio.run(async_example())

    # Run file management example
    file_management_example()
