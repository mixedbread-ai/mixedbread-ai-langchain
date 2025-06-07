"""
Vector Store Chunk Retrieval Example

This example demonstrates how to search for document chunks across vector stores
using the enhanced MixedbreadVectorStoreRetriever with the updated API.
"""
import asyncio
from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever


def basic_chunk_search():
    """Basic chunk search example."""
    print("=== Basic Chunk Search Example ===\n")
    
    # Create a retriever for chunk-based search
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["vector_store_abc123xyz789"],  # Replace with your vector store ID
        top_k=5,
        score_threshold=0.7,
        return_metadata=True,
    )
    
    query = "What is semantic search and how does it work?"
    print(f"Query: {query}")
    
    try:
        documents = retriever.get_relevant_documents(query)
        print(f"Found {len(documents)} relevant chunks:")
        
        for i, doc in enumerate(documents, 1):
            score = doc.metadata.get("relevance_score", 0)
            source = doc.metadata.get("source", "Unknown")
            file_id = doc.metadata.get("file_id", "N/A")
            vector_store_id = doc.metadata.get("vector_store_id", "N/A")
            
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Source: {source}")
            print(f"   File ID: {file_id}")
            print(f"   Vector Store: {vector_store_id}")
            print(f"   Content: {doc.page_content[:150]}...")
            
            # Show additional metadata if available
            if "position" in doc.metadata:
                print(f"   Position: {doc.metadata['position']}")
                
    except Exception as e:
        print(f"Search failed: {e}")
        print("Note: Replace 'vector_store_abc123xyz789' with a valid vector store ID")


def multi_store_search():
    """Multi-store search example."""
    print("\n=== Multi-Store Search Example ===\n")
    
    # Search across multiple vector stores
    multi_retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=[
            "store_1_id",  # Replace with actual vector store IDs
            "store_2_id", 
            "store_3_id"
        ],
        top_k=10,
        score_threshold=0.6,
        return_metadata=True,
    )
    
    queries = [
        "machine learning algorithms",
        "neural network architectures", 
        "natural language processing"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        try:
            results = multi_retriever.get_relevant_documents(query)
            print(f"  Found {len(results)} chunks across {len(multi_retriever.vector_store_ids)} stores")
            
            # Group results by vector store
            store_counts = {}
            for doc in results:
                store_id = doc.metadata.get("vector_store_id", "unknown")
                store_counts[store_id] = store_counts.get(store_id, 0) + 1
            
            for store_id, count in store_counts.items():
                print(f"    {store_id}: {count} chunks")
                
        except Exception as e:
            print(f"  Search failed: {e}")
        print()


def advanced_search_options():
    """Advanced search options example."""
    print("=== Advanced Search Options Example ===\n")
    
    # Retriever with custom search options
    advanced_retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["your_vector_store_id"],  # Replace with your vector store ID
        top_k=8,
        score_threshold=0.75,
        return_metadata=True,
        search_options={
            "return_metadata": True,
            # Add any additional search options supported by the API
        }
    )
    
    print("Configured retriever with:")
    print(f"  - Top K: {advanced_retriever.top_k}")
    print(f"  - Score threshold: {advanced_retriever.score_threshold}")
    print(f"  - Return metadata: {advanced_retriever.return_metadata}")
    print(f"  - Search options: {advanced_retriever.search_options}")
    
    # Dynamic configuration
    print("\nUpdating search options...")
    advanced_retriever.update_search_options(
        score_threshold=0.8,
        return_metadata=True
    )
    
    print(f"  - Updated score threshold: {advanced_retriever.score_threshold}")
    
    # Add/remove vector stores dynamically
    print("\nDynamic vector store management...")
    print(f"  Initial stores: {advanced_retriever.vector_store_ids}")
    
    advanced_retriever.add_vector_store("new_store_id")
    print(f"  After adding store: {advanced_retriever.vector_store_ids}")
    
    advanced_retriever.remove_vector_store("new_store_id")
    print(f"  After removing store: {advanced_retriever.vector_store_ids}")


async def async_search_example():
    """Async search example using direct client access."""
    print("=== Async Search Example ===\n")
    
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["async_vector_store_id"],  # Replace with your vector store ID
        top_k=5,
        score_threshold=0.7
    )
    
    queries = [
        "artificial intelligence applications",
        "deep learning frameworks",
        "computer vision techniques"
    ]
    
    print("Running concurrent searches using direct async client...")
    
    # Method 1: Using convenience method
    async def search_with_convenience_method(query):
        try:
            response = await retriever.search_async(query)
            return query, len(response.data), response.data[:2]
        except Exception as e:
            return query, 0, f"Error: {e}"
    
    # Method 2: Using direct client access
    async def search_with_direct_client(query):
        try:
            response = await retriever.aclient.vector_stores.search(
                query=query,
                vector_store_ids=retriever.vector_store_ids,
                top_k=retriever.top_k,
                score_threshold=retriever.score_threshold
            )
            return query, len(response.data), response.data[:2]
        except Exception as e:
            return query, 0, f"Error: {e}"
    
    print("\n1. Using convenience method (search_async):")
    tasks = [search_with_convenience_method(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for query, count, sample_results in results:
        print(f"\nQuery: {query}")
        if isinstance(sample_results, str):  # Error case
            print(f"  {sample_results}")
        else:
            print(f"  Found {count} chunks")
            for i, chunk in enumerate(sample_results[:2], 1):
                score = getattr(chunk, 'score', 0)
                content = getattr(chunk, 'content', '')
                print(f"    {i}. Score: {score:.3f} | {content[:80]}...")
    
    print("\n2. Using direct client access:")
    tasks = [search_with_direct_client(queries[0])]  # Just test first query
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for query, count, sample_results in results:
        print(f"\nQuery: {query}")
        if isinstance(sample_results, str):  # Error case
            print(f"  {sample_results}")
        else:
            print(f"  Found {count} chunks using direct client access")


def error_handling_example():
    """Error handling example."""
    print("=== Error Handling Example ===\n")
    
    # Example with invalid vector store ID
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["invalid_store_id"],
        top_k=5
    )
    
    print("Testing with invalid vector store ID...")
    try:
        results = retriever.get_relevant_documents("test query")
        print(f"Unexpected success: {len(results)} results")
    except Exception as e:
        print(f"Expected error handled: {type(e).__name__}: {e}")
    
    # Example with empty query
    print("\nTesting with empty query...")
    valid_retriever = MixedbreadVectorStoreRetriever(
        vector_store_ids=["valid_store_id"],  # Replace with valid ID
        top_k=5
    )
    
    results = valid_retriever.get_relevant_documents("")
    print(f"Empty query returned {len(results)} results (expected: 0)")
    
    results = valid_retriever.get_relevant_documents("   ")
    print(f"Whitespace-only query returned {len(results)} results (expected: 0)")


if __name__ == "__main__":
    # Run basic search example
    basic_chunk_search()
    
    # Run multi-store search
    multi_store_search()
    
    # Run advanced options example
    advanced_search_options()
    
    # Run async example
    print("Running async search example...")
    asyncio.run(async_search_example())
    
    # Run error handling example
    error_handling_example()
    
    print("\n=== All examples completed ===")
    print("\nNote: Replace example vector store IDs with your actual vector store IDs to test with real data.")
