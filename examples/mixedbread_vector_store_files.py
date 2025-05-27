from mixedbread_ai_langchain import MixedbreadVectorStoreFileRetriever

retriever = MixedbreadVectorStoreFileRetriever(
    vector_store_ids=["vector_store_abc123xyz789"],
    top_k=3,
    include_chunks=True,
    chunk_limit=2,
)

query = "What is semantic search?"
documents = retriever.get_relevant_documents(query)

print(f"Query: {query}")
print(f"Found {len(documents)} relevant files:")

for i, doc in enumerate(documents, 1):
    score = doc.metadata.get("relevance_score", 0)
    filename = doc.metadata.get("filename", "Unknown")
    file_type = doc.metadata.get("file_type", "Unknown")
    chunks_included = doc.metadata.get("chunks_included", 0)
    total_chunks = doc.metadata.get("total_chunks", 0)

    print(f"\n{i}. Score: {score:.3f}")
    print(f"   File: {filename}")
    print(f"   Type: {file_type}")
    print(f"   Chunks: {chunks_included}/{total_chunks} included")
    print(f"   Content preview: {doc.page_content[:150]}...")

file_only_retriever = MixedbreadVectorStoreFileRetriever(
    vector_store_ids=["vector_store_abc123xyz789"], top_k=5, include_chunks=False
)

file_results = file_only_retriever.get_relevant_documents("artificial intelligence")
print(f"\nFile-only search found {len(file_results)} files")

for i, doc in enumerate(file_results, 1):
    filename = doc.metadata.get("filename", "Unknown")
    score = doc.metadata.get("relevance_score", 0)
    print(f"  {i}. {filename} (Score: {score:.3f})")
