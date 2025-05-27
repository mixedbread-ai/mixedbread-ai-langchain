from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever

retriever = MixedbreadVectorStoreRetriever(
    vector_store_ids=["vector_store_abc123xyz789"],
    top_k=5,
    score_threshold=0.7,
)

query = "What is semantic search?"
documents = retriever.get_relevant_documents(query)

print(f"Query: {query}")
print(f"Found {len(documents)} relevant chunks:")

for i, doc in enumerate(documents, 1):
    score = doc.metadata.get("relevance_score", 0)
    source = doc.metadata.get("source", "Unknown")
    chunk_index = doc.metadata.get("chunk_index", "N/A")

    print(f"\n{i}. Score: {score:.3f}")
    print(f"   Source: {source}")
    print(f"   Chunk: {chunk_index}")
    print(f"   Content: {doc.page_content[:100]}...")

multi_retriever = MixedbreadVectorStoreRetriever(
    vector_store_ids=["store_1_id", "store_2_id", "store_3_id"],
    top_k=10,
    score_threshold=0.6,
)

multi_results = multi_retriever.get_relevant_documents("machine learning")
print(f"\nMulti-store search found {len(multi_results)} chunks")
